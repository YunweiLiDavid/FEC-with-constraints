# Our model is build upon "Image as Set of Points", ICLR23. https://github.com/ma-xu/Context-Cluster/blob/main/models/context_cluster.py
# Thanks the authors for their impressive work!
import os,sys
import copy
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from compute_constraint_loss import orthogonality_loss
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from torch_scatter import scatter_sum

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


class PointReducer(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ClusterPool(nn.Module):
    """
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, stride=4, in_chans=5, embed_dim=64, norm_layer=None, fold_w=1, fold_h=1):
        super().__init__()
        self.norm2 = GroupNorm(embed_dim)
        self.stride = stride
        self.conv_f = nn.Conv2d(in_chans, embed_dim, kernel_size=1)  # for similarity
        self.conv_v = nn.Conv2d(in_chans, embed_dim, kernel_size=1)  # for value
        self.conv_skip = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1, stride=2)  # for skip connection
        self.fold_w = fold_w
        self.fold_h = fold_h

    def forward(self, x):
        identity = self.conv_skip(x)
        value = self.conv_v(x)
        x = self.conv_f(x)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            # only for tasks with very high resolution, e.g., detection
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        
        b, c, w, h = x.shape
        centers = F.adaptive_avg_pool2d(x, (w // self.stride, h // self.stride))
        value_centers = rearrange(F.adaptive_avg_pool2d(value, (w // self.stride, h // self.stride)), 'b c w h -> b (w h) c')
        b, c, ww, hh = centers.shape
        sim = pairwise_cos_sim( centers.reshape(b, c, -1).permute(0, 2, 1), x.reshape(b, c, -1).permute(0, 2, 1) )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        M, N = value_centers.shape[1], value2.shape[1]
        value2 = rearrange(value2, 'b n c -> (b n) c')
        sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')
        idx_offset = (torch.arange(b, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
        sim_max_idx = sim_max_idx + idx_offset
        out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b*M), '(b m) c -> b m c', b=b, m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
        out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        out = rearrange(out, 'b (w h) c -> b c w h', w=ww, h=hh)
        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)

        out = identity + self.norm2(out)
        
        return out


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h

    def forward(self, x):  # [b,c,w,h]
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]

        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]

        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        
        # 1. compute cluster loss
        L_Clst = -torch.mean(sim_max)  
        
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)

        
        # 2. compute separate loss
        mask_neg = 1 - mask  # Mask for non-assigned centers
        sim_neg = sim * mask_neg  # Similarities to non-assigned centers
        # maximum similarity to non-assigned centers
        sim_neg_max, _ = sim_neg.max(dim=1)  # [B,N]
        L_Sep = torch.mean(sim_neg_max)
        
        #3. compute orthogonality loss
        L_Orth = orthogonality_loss(centers)
        '''
        #4. balance regularization
        cluster_probs = mask.sum(dim=-1) / mask.size(-1)  # [B, M]，probability of hard assignment
        L_Entropy = -torch.mean(torch.sum(cluster_probs * torch.log(cluster_probs + 1e-6), dim=-1))
        '''
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        M, N = value_centers.shape[1], value2.shape[1]
        value2 = rearrange(value2, 'b n c -> (b n) c')
        sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')
        idx_offset = (torch.arange(b, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
        sim_max_idx = sim_max_idx + idx_offset
        out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b*M), '(b m) c -> b m c', b=b, m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
        out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        # dispatch step, return to each point in a cluster
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
        out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out, {"L_Orth": L_Orth,  "L_Clst": L_Clst, "L_Sep": L_Sep }


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        out, cluster_losses = self.token_mixer(self.norm1(x))
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * out)
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, cluster_losses


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class FEC(nn.Module):
    """
    feature extraction with clustering (FEC), the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32], **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PointReducer(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=5, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append( ClusterPool(down_stride, embed_dims[i], embed_dims[i + 1], fold_w=fold_w[i+1], fold_h=fold_h[i+1]) )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x

    def forward_tokens(self, x):
        outs = []
        losses = {"L_Sep": 0, "L_Orth": 0, "L_Clst": 0 }
        for idx, block in enumerate(self.network):
            if isinstance(block, torch.nn.Sequential):
                for sub_idx, sub_block in enumerate(block):
                    x, block_losses = sub_block(x)
                    losses["L_Clst"] += block_losses["L_Clst"]
                    losses["L_Sep"] += block_losses["L_Sep"]
                    losses["L_Orth"] += block_losses["L_Orth"]
                    #losses["L_Entropy"] += block_losses["L_Entropy"]
            elif isinstance(block, ClusterPool):
                x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs, losses
        # output only the features of last layer for image classification
        return x, losses

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x, losses = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x, losses
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out, losses


@register_model
def fec_small(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = FEC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def fec_base(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = FEC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def fec_large(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [6, 6, 12, 12]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = FEC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


if has_mmdet:
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class fec_small_seg(FEC):
        def __init__(self, **kwargs):
                layers = [3, 4, 5, 2]
                norm_layer = GroupNorm
                embed_dims = [32, 64, 196, 320]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w = [10, 10, 5, 5]  # modified as the resolution is changed (224 -> 512)
                proposal_h = [10, 10, 5, 5]
                fold_w = [1, 1, 1, 1]
                fold_h = [1, 1, 1, 1]
                heads = [4, 4, 8, 8]
                head_dim = [24, 24, 24, 24]
                down_patch_size = 3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim, fork_feat=True, **kwargs)
    
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class fec_base_seg(FEC):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer = GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w = [10, 10, 5, 5]  # modified as the resolution is changed (224 -> 512)
                proposal_h = [10, 10, 5, 5]
                fold_w = [1, 1, 1, 1]
                fold_h = [1, 1, 1, 1]
                heads = [4, 4, 8, 8]
                head_dim = [32, 32, 32, 32]
                down_patch_size = 3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim, fork_feat=True, **kwargs)
    
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class fec_large_seg(FEC):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer = GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w = [10, 10, 5, 5]  # modified as the resolution is changed (224 -> 512)
                proposal_h = [10, 10, 5, 5]
                fold_w = [1, 1, 1, 1]
                fold_h = [1, 1, 1, 1]
                heads = [6, 6, 12, 12]
                head_dim = [32, 32, 32, 32]
                down_patch_size = 3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim, fork_feat=True, **kwargs)
    
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class fec_small_det(FEC):
        def __init__(self, **kwargs):
                layers = [3, 4, 5, 2]
                norm_layer = GroupNorm
                embed_dims = [32, 64, 196, 320]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w = [7, 7, 7, 7]
                proposal_h = [7, 7, 7, 7]
                fold_w = [2, 2, 1, 1]  # modified as the resolution is changed (224, 224) -> (1333, 800)
                fold_h = [2, 2, 1, 1]  # will OOM w/o fold
                heads = [4, 4, 8, 8]
                head_dim = [24, 24, 24, 24]
                down_patch_size = 3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim, fork_feat=True, **kwargs)
    
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class fec_base_det(FEC):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer = GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w = [7, 7, 7, 7]
                proposal_h = [7, 7, 7, 7]
                fold_w = [2, 2, 1, 1]  # modified as the resolution is changed (224, 224) -> (1333, 800)
                fold_h = [2, 2, 1, 1]  # will OOM w/o fold
                heads = [4, 4, 8, 8]
                head_dim = [32, 32, 32, 32]
                down_patch_size = 3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim, fork_feat=True, **kwargs)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class fec_large_det(FEC):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer = GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w = [7, 7, 7, 7]
                proposal_h = [7, 7, 7, 7]
                fold_w = [2, 2, 1, 1]  # modified as the resolution is changed (224, 224) -> (1333, 800)
                fold_h = [2, 2, 1, 1]  # will OOM w/o fold
                heads = [6, 6, 12, 12]
                head_dim = [32, 32, 32, 32]
                down_patch_size = 3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim, fork_feat=True, **kwargs)


@torch.no_grad()
def compute_throughput(model, batch_size=256, resolution=224):
    import time
    torch.cuda.empty_cache()
    warmup_iters = 20
    num_iters = 100
    device = torch.device('cuda')

    model.eval()
    model.to(device)

    timing = []
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # warmup
    for _ in range(warmup_iters):
        model(inputs)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    return (batch_size / timing.mean()).item()


def get_flops0():
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        net = fec_small()
        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def get_flops1():
    from collections import Counter
    import numpy as np
    def fvcore_mul_flop_jit(inputs, outputs):
        flop_dict = Counter()
        flop_dict["mul"] = np.prod(inputs[0].type().sizes())
        return flop_dict
    
    input = torch.rand(1, 3, 224, 224)
    model = fec_small()
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input)
    
    # flops.set_op_handle(**{'aten::mul': fvcore_mul_flop_jit, 'aten::div': fvcore_mul_flop_jit, 'aten::mul_': fvcore_mul_flop_jit, 'aten::add': fvcore_mul_flop_jit, 'aten::sum': fvcore_mul_flop_jit, 'aten::mean': fvcore_mul_flop_jit, 'aten::sub': fvcore_mul_flop_jit, 'aten::scatter_': fvcore_mul_flop_jit})

    print("FLOPs: ", flops.total()/10.**9)


if __name__ == '__main__':
    input = torch.rand(32, 3, 224, 224)
    model = fec_small()
    out, losses = model(input)
    print(model)
    print(out.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters/1024**2))

    for i in range(3):
        print(compute_throughput(model), end=' ')

    get_flops1()

import torch
import torch.nn.functional as F
def compute_cls_loss(features, centers, labels):
    features = F.normalize(features, dim=-1)
    centers = F.normalize(centers, dim=-1)
    assigned_centers = centers[labels]
    cos_sim = torch.sum(features * assigned_centers, dim=-1)
    return -torch.mean(cos_sim)

def compute_sep_loss(features, centers, labels):
    features = F.normalize(features, dim=-1)
    centers = F.normalize(centers, dim=-1)
    cos_sim_all = torch.matmul(features, centers.T)
    mask = torch.arange(len(centers), device=features.device) != labels.unsqueeze(-1)
    non_target_sim = cos_sim_all[mask].view(len(features), -1)
    return torch.mean(torch.max(non_target_sim, dim=1)[0])

def compute_orth_loss(centers):
    centers = F.normalize(centers, dim=-1)  # 归一化
    gram_matrix = torch.matmul(centers, centers.T)  # Gram 矩阵 [M, M]
    identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)  # 单位矩阵
    return torch.norm(gram_matrix - identity, p='fro') ** 2  # Frobenius 范数的平方

   
def orthogonality_loss(centers):
    """
    centers [b,c,C_W,C_H]
    """   
    L_Orth = 0
    for b in range(centers.size(0)):  # 遍历每个样本
        L_Orth += compute_orth_loss(centers[b])  # 对每个样本的聚类中心计算
    L_Orth /= centers.size(0)  # 平均损失

    return L_Orth
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
    b, c, C_W, C_H = centers.size()
    M = C_W * C_H  # 聚类中心的数量

    # 将 centers 展平为 [b, c, M]
    centers = centers.view(b, c, M)

    # 对每个样本的每个聚类中心进行归一化
    centers = F.normalize(centers, dim=-1)

    # 计算 Gram 矩阵 [b, M, M]
    gram_matrices = torch.bmm(centers.transpose(1, 2), centers)

    # 创建单位矩阵 [b, M, M]
    identity = torch.eye(M, device=centers.device).unsqueeze(0).expand(b, -1, -1)

    # 计算损失：Gram 矩阵与单位矩阵的差异
    loss = torch.norm(gram_matrices - identity, p='fro', dim=(1, 2)) ** 2  # 按样本计算 Frobenius 范数的平方
    L_Orth = loss.mean()  # 计算平均损失

    return L_Orth
import torch
import torch.nn as nn
import torch.nn.functional as F
class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1.0):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        '''
        W: [B, C, N]
        '''
        B, p, m = W.shape
        I = torch.eye(p,device=W.device).expand(B, p, p) + 1e-6 * torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.linalg.slogdet(I + scalar * W.matmul(W.transpose(-1, -2)))[1]
        if torch.isnan(logdet).any():
            nan_indices = torch.where(torch.isnan(logdet))  # 找到 logdet 变成 nan 的索引
            print(f"NaN detected in logdet at indices: {nan_indices}")
            print(f"W.shape: {W.shape}, scalar: {scalar}, I.shape: {I.shape}")
            print(f"W.matmul(W.T) at NaN indices: {W[nan_indices[0]] @ W[nan_indices[0]].transpose(-1, -2)}")
        return logdet.mean() / 2.  #  batch mean loss
    
    def compute_compress_loss(self, W, Pi):
        '''
        p: number of channels
        m: number of points
        k: number of clusters
        '''
        B, p, m = W.shape

        B, k, _, _ = Pi.shape

        I = torch.eye(p,device=W.device).expand((B,k,p,p)) + 1e-6 * torch.eye(p, device=W.device)
        trPi = Pi.sum(3) + 1e-8

        scale = (p/(trPi*self.eps)).view(B,k,1,1)
        
        W = W.unsqueeze(1) #[B, 1, p, m]
        log_det = torch.linalg.slogdet(I + scale*W.mul(Pi).matmul(W.transpose(-1,-2)))[1]

        compress_loss = (trPi.squeeze()*log_det/(2*m)).sum()
        #print(compress_loss)
        return compress_loss
        
    def forward(self, X, mask):
        #This function support Y as label integer or membership probablity.
        '''
        M: number of clusters
        N: number of points
        '''
        '''
        if len(Y.shape)==1:
            #if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes,1,Y.shape[0]),device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label,0,indx] = 1
        else:
            #if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes,1,-1))
        '''    
        B, C, W, H = X.shape
        W = X.view(B, C, -1)  # [B, C, W, H] -> [B, C, W*H]
        # compute Pi
        Pi = mask  #  [B, M, N]
        Pi = Pi.unsqueeze(2)  #  [B, M, 1, N]
        
        discrimn_loss = self.compute_discrimn_loss(W)

        compress_loss = self.compute_compress_loss(W, Pi)
        total_loss = - discrimn_loss + self.gamma*compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]
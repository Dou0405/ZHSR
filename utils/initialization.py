"""SPA endmember initialization from the observed LR only."""
import torch

def extract_spa(X, R):
    """ SPA: 连续投影算法 (极速版) """
    device = X.device
    X = X.float()
    bands, N = X.shape
    
    indices = []
    X_proj = X.clone()
    
    for i in range(R):
        # 1. 寻找当前投影空间中能量(L2范数)最大的像素
        norms = torch.norm(X_proj, dim=0)
        idx = torch.argmax(norms).item()
        indices.append(idx)
        
        # 2. 选定为端元
        e = X_proj[:, idx:idx+1] 
        
        # 3. 将所有数据投影到该端元的正交子空间中
        e_norm_sq = torch.sum(e**2) + 1e-8
        proj_coeff = torch.matmul(e.t(), X_proj) / e_norm_sq
        X_proj = X_proj - torch.matmul(e, proj_coeff)
        
    return X[:, indices], indices

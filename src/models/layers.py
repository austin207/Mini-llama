import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root mean square layer normalization"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim = -1, keepdim = True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)  # Critical fix

    def forward(self, x):
        return self.out_proj(F.silu(self.w1(x)) * self.w2(x))
    

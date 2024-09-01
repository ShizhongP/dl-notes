
import torch
import torch.nn as nn

'''
RNSNorm: w *   x / RMS(x) 
RMS(x) = sqrt( mean(x^2) + eps)
'''
class RNSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float =1e-6):
        super(RNSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def _norm(self,x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
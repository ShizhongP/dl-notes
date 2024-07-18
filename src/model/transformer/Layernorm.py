import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.gamma = nn.Parameter(torch.ones(self.hidden_size))
        self.beta = nn.Parameter(torch.zeros(self.hidden_size))
        self.eps = eps
        
    def forward(self, x):
        # x : batch_size, sequence _length, hidden_size
        mean = x.mean(-1, keepdim=True) # batch_size, sequence _length , 1
        std = x.std(-1, keepdim=True)   # batch_size, sequence _length , 1
        
        # this applis Broadcast mechanism in the last dimension
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
def test():
    batch_size=3    
    sequence_length= 4
    hidden_size = 4
    
    x = torch.randn(batch_size, sequence_length, hidden_size)
    layer_norm = LayerNorm(hidden_size)
    
    x_norm = layer_norm(x)
    print(x)
    print(x_norm)
    
if __name__ =="__main__":
    test()
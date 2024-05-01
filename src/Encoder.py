import torch
import torch.nn as nn 

from EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers, dim_embedding, num_heads, dim_ff=None, dropout=0.1):
        super(Encoder,self).__init__()
        
        self.layers = nn.ModuleList([EncoderLayer(dim_embedding, num_heads, dim_ff, dropout) for _ in range(num_layers)])
        
    def forward(self,x,mask=None):
        # x: batch_size, sequence_length, dim_embedding
        
        for layer in self.layers:
            x = layer(x,mask)

        return x
    
def test():
    dim_embedding = 16
    batch_size = 3
    sequence_length = 4 
    dim_ff =16
    
    x = torch.rand(batch_size, sequence_length, dim_embedding)
    
    encoder = Encoder(num_layers=6, dim_embedding=dim_embedding, num_heads=8, dim_ff=dim_ff, dropout=0.1)

    output = encoder(x)
    print(output.shape)

if __name__ =="__main__":
    test()
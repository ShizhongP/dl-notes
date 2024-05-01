import torch
import torch.nn as nn

from DecoderLayer import DecoderLayer
from Encoder import Encoder
class Decoder(nn.Module):
    def __init__(self,dim_embedding,num_layers,num_heads,dim_ff,dropout):
        super(Decoder,self).__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(dim_embedding,dim_ff,num_heads,dropout) for _ in range(num_layers)])
        
    def forward(self,x,encode_output):
        # x: batch_size, sequence_length, dim_embedding
        for layer in self.layers:
            x = layer(x,encode_output)
            
        return x
    
def test():
    dim_embedding = 16
    batch_size = 3
    sequence_length = 4
    dim_ff = 16
    num_heads = 8
    
    x = torch.rand(batch_size,sequence_length,dim_embedding)
    
    encoder = Encoder(num_layers=6,dim_embedding=dim_embedding,num_heads=num_heads,dim_ff=dim_ff,dropout=0.1)
    decoder = Decoder(dim_embedding=dim_embedding,num_layers=6,num_heads=num_heads,dim_ff=dim_ff,dropout=0.1)
    
    encode_output = encoder(x)
    output = decoder(x,encode_output)
    
    print(output.shape)
    
if __name__ =="__main__":
    test()
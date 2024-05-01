import torch 
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder
from Embedding import TransFormerEmbedding

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim_embedding, num_layers, num_heads, dim_ff, dropout):
        super(Transformer, self).__init__()
        
        self.embedding = TransFormerEmbedding(vocab_size, dim_embedding)
        self.encoder = Encoder(dim_embedding,num_layers,  num_heads, dim_ff, dropout)
        self.decoder = Decoder(dim_embedding, num_layers, num_heads, dim_ff, dropout)
        
    def forward(self, src, trg):
        # x: batch_size, sequence_length
        x = self.embedding(src)
        encode_output = self.encoder(x)
        y = self.embedding(trg)
        output = self.decoder(y, encode_output)
        
        return output
    
def test():
    batch_size = 3
    sequence_length = 4
    vocab_size = 200
    dim_embedding = 16
    num_layers = 6
    num_heads = 8
    dim_ff = 16
    dropout = 0.1
    
    x = torch.ones(batch_size, sequence_length, dtype=torch.long)
    y = torch.ones(batch_size, sequence_length,  dtype=torch.long)
    transformer = Transformer(vocab_size, dim_embedding, num_layers, num_heads, dim_ff, dropout)
    
    output = transformer(x,y)
    
    print(output.shape)
    
if __name__ =="__main__":
    test()
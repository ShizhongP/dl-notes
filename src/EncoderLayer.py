import torch
import torch.nn as nn
from Layernorm  import LayerNorm
from MultiHeadAttention import MultiHeadAttention
from PositionFeedForward import PositionFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, dim_embedding, num_heads, dim_ff=None, dropout=0.1):
        super(EncoderLayer,self ).__init__()
        
        if dim_ff is None:
            dim_ff = dim_embedding
            
        self.attention = MultiHeadAttention(dim_embedding, dim_qk=dim_embedding, dim_v=dim_embedding,)
        self.position_feedforward = PositionFeedForward(dim_embedding, dim_ff, dropout=dropout)
        self.layer_norm_1 = LayerNorm(dim_embedding)
        self.layer_norm_2 = LayerNorm(dim_embedding)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,mask=None):
        # x: batch_size, sequence_length ,dim_embedding
        
        # multihead attention
        atten_socre ,_= self.attention(x,mask) # batch_size, sequence_length, dim_embedding
        
        # Add and normalization
        y  = self.layer_norm_1(self.dropout(atten_socre) + x)
        
        # position feedforward
        feed_forward = self.position_feedforward(y) # batch_size, sequence_length, dim_embedding
        
        # Add and normalization
        output = self.layer_norm_2(self.dropout(feed_forward) + y)
        
        return output
    
def test():
    batch_size = 3
    sequence_length = 4
    dim_embedding = 16
    dim_ff = 16
    
    encoder_layer = EncoderLayer(dim_embedding=dim_embedding, num_heads=8, dim_ff=dim_ff, dropout=0.1)
    
    x = torch.rand(batch_size, sequence_length, dim_embedding)
    
    output = encoder_layer(x)
    
    print(output.shape)
    
if __name__ == "__main__":
    test()
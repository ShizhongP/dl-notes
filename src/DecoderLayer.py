import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from PositionFeedForward import PositionFeedForward
from Layernorm import LayerNorm
from Encoder import Encoder
class DecoderLayer(nn.Module):
    def __init__(self,dim_embedding,dim_ff,num_heads=8,dropout=0.1):
        super(DecoderLayer,self).__init__()
        
        self.attention_1 = MultiHeadAttention(dim_embedding,dim_embedding,dim_embedding,num_heads=num_heads)
        self.attention_2 = MultiHeadAttention(dim_embedding,dim_embedding,dim_embedding,num_heads=num_heads)
        self.position_feedforward = PositionFeedForward(dim_embedding,dim_ff,dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = LayerNorm(dim_embedding)
        self.layer_norm_2 = LayerNorm(dim_embedding)
        self.layer_norm_3 = LayerNorm(dim_embedding)
        
    def forward(self,x,encode_output):
        
        # x : batch_size, sequence_length, dim_embedding
        # self-attention
        self_atten, atten_weitght = self.attention_1(x)
        # add and normalization
        y = self.layer_norm_1(self.dropout(self_atten) + x)
        
        # encode-attention
        encode_attention,_ =  self.attention_2(encode_output)
        # add and normalization
        y = self.layer_norm_2(self.dropout(encode_attention) + y)
        
        # position feedforward
        feed_forward = self.position_feedforward(y)
        # add and normalization
        output = self.layer_norm_3(self.dropout(feed_forward) + y)
        
        return output
    
def test():
    dim_embedding = 16
    batch_size = 3
    sequence_length = 4
    dim_ff = 16
    num_heads = 8
    
    x = torch.rand(batch_size,sequence_length,dim_embedding)
    decoder_layer = DecoderLayer(dim_embedding,dim_ff,num_heads=num_heads,dropout=0.1)
    encoder = Encoder(num_layers=6,dim_embedding=dim_embedding,num_heads=num_heads,dim_ff=dim_ff,dropout=0.1)
    
    encode_output = encoder(x)
    output = decoder_layer(x, encode_output)
    
    print(output.shape)
    
if __name__ == "__main__":
    test()
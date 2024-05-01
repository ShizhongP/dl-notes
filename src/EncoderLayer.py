import torch
import torch.nn
from Layernorm  import LayerNorm
from MultiheadAttention import MultiHeadAttention
from PositionFeedforward import PositionFeedforward


class EncoderLayer(nn.Module):
    def __init__(self, ):
        super(EncoderLayer, dim_embedding, dim_qk,dim_v, num_heads, dim_ff=None, dropout=0.1).__init__()
        
        if dim_ff is None:
            dim_ff = dim_embedding
            
        self.attention = MultiHeadAttention(dim_embedding, dim_qk, dim_v,)
        self.position_feedforward = PositionFeedforward(dim_embedding, dim_ff, dropout=dropout)
        self.layer_norm_1 = LayerNorm(dim_embedding)
        self.layer_norm_2 = LayerNorm(dim_embedding)
        self.dropout = nn.Dropout(dropout)
    
    def forwar(self,x):
        # x: batch_size, sequence_length ,dim_embedding
        
        # multihead attention
        atten_socre = self.attention(x)
        
        # Add and normalization
        y  = self.layer_norm_1(self.dropout(atten_socre) + x)
        
        # position feedforward
        feed_forward = self.position_feedforward(y)
        
        # Add and normalization
        output = self.layer_norm_2(self.dropout(feed_forward) + y)
        
        return output
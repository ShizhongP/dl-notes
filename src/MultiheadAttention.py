import torch
import torch.nn as nn
from math import sqrt

class MultiheadAttention(nn.Module):
    
    def __init__(self,dim_embedding, dim_qk,dim_v,num_heads=8):
        super(MultiheadAttention, self).__init__()
        
        self.dim_embedding=dim_embedding
        self.dim_qk=dim_qk
        self.num_heads=num_heads
        self.dim_v=dim_v
        
        self._normal_factor =  (self.dim_qk//num_heads) ** -0.5
        
        self.linear_q=nn.Linear(dim_embedding,dim_qk,bias=False)
        self.linear_k=nn.Linear(dim_embedding,dim_qk,bias=False)
        
        self.linear_v=nn.Linear(dim_embedding,dim_v,bias=False)
        
        
    
    def forward(self,x):
        batch_size,sequence_length, dim_embedding = x.shape
        
        assert self.dim_embedding == dim_embedding
        
        div_k = self.dim_qk // self.num_heads
        div_v = self.dim_v // self.num_heads
        
        # batch_size , num_heads , sequence_length , div_k
        k =self.linear_k(x).reshape(batch_size,sequence_length,self.num_heads,div_k).transpose(1,2)
        q =self.linear_q(x).reshape(batch_size,sequence_length,self.num_heads,div_k).transpose(1,2)
        # batch_size , num_heads , sequence_length , div_v
        v =self.linear_v(x).reshape(batch_size,sequence_length,self.num_heads,div_v).transpose(1,2)
         
        dist = torch.matmul(q,k.transpose(-2,-1)) * self._normal_factor
        dist = torch.softmax(dist,dim=-1)
        
        atten_score = torch.matmul(dist,v) # batch_size, num_heads, sequence_length, div
        atten_score = atten_score.transpose(1,2) # batch_size, sequence_length , num_heads, div
        atten_score = atten_score.reshape(batch_size,sequence_length,self.num_heads*div_v) 
        
        return atten_score 


class MultiHeadAttention(nn.Module):
    
    def __init__(self,dim_embedding, dim_qk, dim_v , num_heads=8,dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dropout = dropout
        
        # self._normal_factor = 1/sqrt(self.dim_qk//num_heads)
        self._normal_factor =  (self.dim_qk // self.num_heads) ** -0.5
        
        self.linear_q = nn.Linear(dim_embedding,dim_qk, bias=False) 
        self.linear_k = nn.Linear(dim_embedding,dim_qk,bias=False)
        self.linear_v = nn.Linear(dim_embedding,dim_v,bias=False)
        
        if  dropout is not None:
            self.dropout = nn.Dropout(dropout)
            
    def forward(self,x,mask=None):
        
        batch_size, sequence_length, dim_embedding = x.shape
        
        assert self.dim_embedding == dim_embedding
        
        div_k = self.dim_qk // self.num_heads
        div_v = self.dim_v // self.num_heads
        
        # divided into mutil heads
        q = self.linear_q(x).view(batch_size,sequence_length,self.num_heads,div_k).permute(0,2,1,3)
        k = self.linear_k(x).view(batch_size,sequence_length,self.num_heads,div_k).permute(0,2,1,3)
        v = self.linear_v(x).view(batch_size,sequence_length,self.num_heads,div_v).permute(0,2,1,3)
        
        
        # Combine queries and keys
        logits = torch.matmul(q, k.permute(0, 1, 3, 2)) * self._normal_factor

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads

        # Convert to probabilites
        weights = torch.softmax(logits, dim=-1)

        # Dropout
        if self.dropout is not None:
            weights = self.dropout(weights)

        atten_score = torch.matmul(weights, v)
        
        atten_score = atten_score.permute(0,2,1,3).contiguous().view(batch_size,sequence_length,self.num_heads*div_v)

        return atten_score, attetion_weights


def test():
    batch_size = 3    # 每次选取3句话
    dim_embedding = 16    #input_size(embedding维度(正常来说是512，为了方便设置为16)。后面8个头的话，要进行截断。2个维度一份，一共8份)
    sequence_length = 4    #每句话固定有4个单词(单词之间计算注意力)
    num_heads = 8    # 8个头
    dim_v = 16   #V向量的长度(V向量长度可以与Q and K不一样)
    dim_qk = 16    #Q and K向量的长度(dim_embedding经过Wq、Wk变换后QK向量长度变为dim_QK)

    x = torch.randn(batch_size,sequence_length,dim_embedding)
    
    multiattention=MultiHeadAttention(dim_embedding,dim_qk,dim_v, num_heads=8)
    att = multiattention(x)
    print(att[0].shape)
    
    multiattention2=MultiheadAttention(dim_embedding,dim_qk,dim_v, num_heads=8)
    att2 = multiattention2(x)
    
    print(att2.shape)
    from pprint import pprint
    pprint(att[0])
    pprint(att2)
test()
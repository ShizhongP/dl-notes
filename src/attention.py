import torch
import torch.nn as nn
from math import sqrt

'''
    input_shape : batch_size, sequence_length, embedding_size
    
    output batch_size, sequence_length, dim_v
'''

class SelfAttention(nn.Module):
    def __init__(self,dim_embedding,dim_qk,dim_v):
        super(SelfAttention, self).__init__()
        
        self.dim_embedding=dim_embedding
        self.dim_qk=dim_qk
        self.dim_v=dim_v
        
        self.linear_q=nn.Linear(dim_embedding,dim_qk,bias=False)
        self.linear_k=nn.Linear(dim_embedding,dim_qk,bias=False)
        
        self.linear_v=nn.Linear(dim_embedding,dim_v,bias=False)
        
        self.normal_factor = 1/sqrt(dim_embedding)
        
    def forward(self,x):
        batch_size,sequence_length,dim_embedding = x.shape
        
        assert self.dim_embedding == dim_embedding
        
        q = self.linear_q(x)
        print(f"Q shape:{q.shape}")
        k = self.linear_k(x)
        print(f"K shape:{k.shape}")
        v = self.linear_v(x)
        print(f"V shape:{v.shape}")
        k_t = k.transpose(1,2)
        print(f"k_t shape:{k_t.shape}")
        dist = torch.bmm(q, k_t) * self.normal_factor
        print(f"dist shape:{dist.shape}") 
        dist = torch.softmax(dist,dim=-1)  
        print(f"dist shape:{dist.shape}")
        atten_score = torch.bmm(dist,v)
        print(f"atten_score shape:{atten_score.shape}")
        return atten_score
        

def test():
    # Inputs to the attention module
    batch_size = 3    # 每次选取3句话
    dim_embedding = 6    #input_size
    sequence_length = 4    #每句话固定有4个单词(单词之间计算注意力)
    dim_V = 8    #V向量的长度(V向量长度可以与Q and K不一样)
    dim_QK = 7    #Q and K向量的长度(dim_embedding经过Wq、Wk变换后QK向量长度变为dim_QK)

    print(f"paramaters:\nbatch_size: {batch_size}\n"
          f"dim_embedding: {dim_embedding}\n"
          f"sequence_length: {sequence_length}\n"
          f"dim_V: {dim_V}\n"
          f"dim_QK: {dim_QK}\n")
    # 输入的数据
    x_gen = torch.randn(batch_size, sequence_length, dim_embedding)
    
    print(f"Input shape :{x_gen.shape}")
    attention = SelfAttention(dim_embedding, dim_QK, dim_V)
    att = attention(x_gen)
test()
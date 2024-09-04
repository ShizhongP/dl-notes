import torch
import torch.nn as nn
import config
# Transformer Standard Attention with kv cache
class Transformer_Attention(nn.Module):
    def __init__(self, dim_embeding:int,dim_qk:int,dim_v:int,kv_cache:bool=True):
        super(Transformer_Attention, self).__init__()
        self.dim_embeding=dim_embeding
        self.dim_qk=dim_qk
        self.dim_v=dim_v
        
        self.linear_q=nn.Linear(dim_embeding,dim_qk,bias=False)
        self.linear_k=nn.Linear(dim_embeding,dim_qk,bias=False)
        self.linear_v=nn.Linear(dim_embeding,dim_v,bias=False)
        
        self._normal_factor =  (self.dim_qk//2) ** -0.5
        
        self.kv_cache=kv_cache
        if kv_cache:
            self.kv_cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, dim_qk))
            self.kv_cache_v = torch.zeros((config.max_batch_size,config.max_seq_len, dim_v))
        
        self.output = nn.Linear(dim_v,dim_embeding,bias=False)
        
    def forward(self,x):
        batch_size,sequence_length, dim_embedding = x.shape
        
        assert self.dim_embeding == dim_embedding
        
        # batch_size , sequence_length , dim_qk
        k =self.linear_k(x)
        q =self.linear_q(x)
        # batch_size , sequence_length , dim_v
        v =self.linear_v(x)
        
        if self.kv_cache:
            k = torch.cat([self.kv_cache_k,k],dim=1)
            v = torch.cat([self.kv_cache_v,v],dim=1)
            self.kv_cache_k = k[:,-1:].detach()
            self.kv_cache_v = v[:,-1:].detach()
        
        dist = torch.matmul(q,k.transpose(-2,-1)) * self._normal_factor
        
        weight = dist.detach()
        dist = torch.softmax(dist,dim=-1)
        
        atten_score = torch.matmul(dist,v)
        atten_score = self.output(atten_score)
        
        return atten_score, weight

# Llama2 Attention with kv-cache and  GQA
class Attention(nn.Module):
    def __init__(self, dim_embedding:int,dim_qk:int,dim_v:int,kv_cache:bool=True):
        super(Attention, self).__init__()
        self.dim_embedding=dim_embedding
        self.dim_qk=dim_qk
        self.dim_v=dim_v
        
        self.linear_q=nn.Linear(dim_embedding,dim_qk,bias=False)
        self.linear_k=nn.Linear(dim_embedding,dim_qk,bias=False)
        self.linear_v=nn.Linear(dim_embedding,dim_v,bias=False)
        
        self._normal_factor =  (self.dim_qk//2) ** -0.5
        
        self.kv_cache=kv_cache
        if kv_cache:
            self.kv_cache_k = torch.zeros((1, 1, dim_qk))
            self.kv_cache_v = torch.zeros((1, 1, dim_v))
        
        self.output = nn.Linear(dim_v,dim_embedding,bias=False)
    def forward(self,x):
        pass
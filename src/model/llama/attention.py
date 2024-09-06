import torch
import torch.nn as nn
import config
from typing import Optional,Tuple
import math
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

# Transformer Standard Attention with kv cache


class Transformer_Attention(nn.Module):
    def __init__(self, dim_embeding: int, dim_qk: int, dim_v: int, kv_cache: bool = True):
        super(Transformer_Attention, self).__init__()
        self.dim_embeding = dim_embeding
        self.dim_qk = dim_qk
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_embeding, dim_qk, bias=False)
        self.linear_k = nn.Linear(dim_embeding, dim_qk, bias=False)
        self.linear_v = nn.Linear(dim_embeding, dim_v, bias=False)

        self._normal_factor = (self.dim_qk//2) ** -0.5

        self.kv_cache = kv_cache
        if kv_cache:
            self.kv_cache_k = torch.zeros(
                (config.max_batch_size, config.max_seq_len, dim_qk))
            self.kv_cache_v = torch.zeros(
                (config.max_batch_size, config.max_seq_len, dim_v))

        self.output = nn.Linear(dim_v, dim_embeding, bias=False)

    def forward(self, x, start_pos, mask):
        bs, seq_len, dim_embedding = x.shape

        assert self.dim_embeding == dim_embedding

        # batch_size , sequence_length , dim_qk
        k = self.linear_k(x)
        q = self.linear_q(x)
        # batch_size , sequence_length , dim_v
        v = self.linear_v(x)

        if self.kv_cache:
            self.kv_cache_k[:bs, start_pos:start_pos+seq_len] = k
            self.kv_cache_v[:bs, start_pos:start_pos+seq_len] = v

            k = self.kv_cache_k[:bs, :start_pos+seq_len]
            v = self.kv_cache_v[:bs, :start_pos+seq_len]

        dist = torch.matmul(q, k.transpose(-2, -1)) * self._normal_factor
        weight = dist.detach()

        if mask:
            dist = dist + mask
            # dist = dist.masked_fill(mask, -1e9)
        dist = torch.softmax(dist, dim=-1)

        atten_score = torch.matmul(dist, v)
        atten_score = self.output(atten_score)

        return atten_score, weight


# MHA with kv-cache
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_em, dim_qk, dim_v, num_heads=8, kv_cache=True):
        super(MultiHeadAttention, self).__init__()
        self.dim_em = dim_em
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.kv_cache = kv_cache

        self.model_parallel_size = fs_init.get_model_parallel_world_size()

        self.head_dim = dim_em // num_heads
        self.local_heads = num_heads // self.model_parallel_size

        self.liner_q = ColumnParallelLinear(
            dim_em, dim_qk*self.num_heads, bias=False, gather_output=False, init_method=lambda x: x)
        self.linear_k = ColumnParallelLinear(
            dim_em, dim_qk*self.num_heads, bias=False, gather_output=False, init_method=lambda x: x)
        self.linear_v = ColumnParallelLinear(
            dim_em, dim_v*self.num_heads, bias=False, gather_output=False, init_method=lambda x: x)
        self.linear_out = RowParallelLinear(
            dim_v*self.num_heads, dim_em, bias=False, gather_output=True, init_method=lambda x: x)
        
        if kv_cache:
            self.kv_cache_k = torch.zeros(
                (config.max_batch_size, config.max_seq_len, self.local_heads, dim_qk))
            self.kv_cache_v = torch.zeros(
                (config.max_batch_size, config.max_seq_len, self.local_heads, dim_v))
            
    
    def forward(self,x,start_pos,mask):
        bs, seq_len, dim_em = x.shape
        
        # bs, seq_len 
        xq =  self.liner_q(x)
        xk = self.linear_k(x)
        xv = self.linear_v(x)
        
        xq = xq.view(bs,seq_len,self.local_heads,self.dim_qk)
        xk = xk.view(bs,seq_len,self.local_heads,self.dim_qk)
        xv = xv.view(bs,seq_len,self.local_heads,self.dim_v)
        
        if self.kv_cache:
            self.kv_cache_k[:bs, start_pos:start_pos+seq_len] = xk
            self.kv_cache_v[:bs, start_pos:start_pos+seq_len] = xv
            
            # bs,cache_len + seq_len, local_heads,dim_qk
            xk = self.kv_cache_k[:bs, :start_pos+seq_len]
            # bs,cache_len + seq_len, local_heads, dim_v
            xv = self.kv_cache_v[:bs, :start_pos+seq_len]
        
        xq = xq.transpose(1,2) # bs,local_heads,seq_len,dim_qk
        keys = xk.transpose(1,2)# bs,local_heads,cache_len+seq_len,dim_qk
        values = xv.transpose(1,2) # bs,local_heads,cache_len+seq_len,dim_v
        
        # bs,local_heads,seq_len,cache_len+seq_len 
        dist = torch.matmul(xq,keys.transpose(2,3)) * (self.dim_qk ** -0.5)

        if mask:
            dist = dist + mask
        weight = dist.detach()
        
        attention_score = torch.softmax(dist,dim=-1)
        attention_score = torch.matmul(attention_score,values)
        output = self.linear_out(attention_score)
        
        return weight,output
        
# Llama2 Attention with kv-cache and  GQA
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
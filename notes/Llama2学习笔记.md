参考博客

[知乎llama2结构详解](https://zhuanlan.zhihu.com/p/649756898)

## Llama2 

### 模型结构

<img src="https://pic4.zhimg.com/80/v2-c9b10194c5e0aa9777afa984063e7ff3_720w.webp" alt="img" style="zoom: 67%;" />

Llama 2的模型结构与标准的Transformer Decoder结构基本一致，主要由32个 Transformer Block 组成，不同之处主要包括以下几点：

1. 前置的**RMSNorm**层
2. Q在与K相乘之前，先使用**RoPE**进行位置编码
3. **K V Cache**，并采用**Group Query Attention**
4. FeedForward层

那么下文将结合具体的代码来展开聊一聊这些差异

#### RMSNorm

Transformer的Normalization层使用的是LayerNormlization层归一化对Tensor进行归一化

**RMSNorm**是LayerNormlization的变体，它省去了求均值的过程，也没有偏置  



#### RoPE旋转位置编码

https://zhuanlan.zhihu.com/p/642884818

旋转位置编码，使在计算attention的时候，也够考虑相对位置信息

#### KV cache && GQA （分组查询注意力机制）

[kv cache](https://zhuanlan.zhihu.com/p/630832593)

[MHA, MQA, GQA](https://mp.weixin.qq.com/s/_4OxoRLxhOcjGf0Q4Tvp2Q)




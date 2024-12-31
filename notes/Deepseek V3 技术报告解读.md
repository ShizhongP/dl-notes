# Deepseek V3 技术报告解读

## 概述

参数量：  671B,  每个token的激活值：37B

架构： MOE（Mixture-of-Experts），MLA（Multi-head Latent Attention）

训练数据量：14.8 trillion ，训练时长：2.788M H800 GPU 小时

训练策略与算法： auxiliary-loss-free，multi-token-prediction training

训练加速和优化: FP8混精度训练，DualPipe，高效跨节点 all-to-all 通信内核

训练成本:

<img src="/home/psz/workspace/AI/My-project/dl-notes/notes/assets/image-20241229170946682.png" alt="image-20241229170946682" style="zoom:50%;" />

下面按照论文，详细解释每一部分到底做了什么

## 架构

### MLA

常规的点积Attention的计算公式为 $o = softmax(\frac{qk^T}{\sqrt{d}})v$​

多头点积Attention（MHA)的计算公式为: $o^{(i)}=softmax(\frac{q^{(i)}k^{(i)T}}{\sqrt{d_h}})v^{(i)}$ , $o = [ o^{(1)};o^{(2)};...o^{(n)}]$

其中 $1 \le i \le n $ , $n$ 为注意力头的个数, 则 $d_i$ 为每个头的隐藏层大小, `;`为concat操作

### MOE



### Auxiliay-loss-free

通过动态调整每个expert的门控值来实现负载均衡



### MTP 





# RAG

## RAG for LLM :  A survey

RAG提出的目的，解决以下问题

1. 大模型幻觉问题
2. 数据新鲜程度问题
3. 数据隐私问题

有以下几种形式

1. Navie RAG
2. Advanced RAG
3. Moduler RAG

评估，和指标

1. 检索的质量
    - Hit Rate
    - MRR
    - NDCG
2. 生成的质量

未来的发展方向

1. RAG or Long-context
    - 长文本需要大量计算，所以仍然需要RAG,可以聚焦于如何使用RAG解决超长文本的问题
2. 多模态的RAG
    - 图像，音频，代码等
3. 继续提高RAG的健壮性，提升准确性和相关性

# 模型轻量化

## Light-PEFT

### motivation

高效微调领域，目前的方法存在以下局限

1. 量化但是没有减少参数的数量
2. 减少参数但是聚焦于推理没有减少训练时的参数
3. Lora,Qlora虽然减少了训练的的参数，但是可训练矩阵的秩是不变的，不是动态决定的
4. 有一些方法能够动态决定可学习模块的大小，但在训练过程中需要持续评估

### abstract

Lig-PEFT提出的在训练早期评估可学习模块的大小，使得最终需要的训练时间减少

### Points

1. 训练早期的过程加入掩码向量,对掩码向量进行训练，决定不同神经元,attention头的权重，
2. 可学习模块的裁剪
    - 粗粒度：较少不必要的可学习模块的数量
    - 细粒度：根据一阶泰勒展开，减少原始大小的增量矩阵的某一行，即下投影和上投影中间隐藏层的大小

### Defect

1. 在训练之前，难以决定那一步骤开始进行剪枝
2. 只有单任务场景的剪枝，缺少多任务的实验

个人认为，损失是还有一点点大

### Pruning LargeLanguage Models to Intra-module Low-rank Architecture with Transitional Activations

waitting to update
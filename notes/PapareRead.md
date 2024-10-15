
# 长文本

## ACL2024: LongBench

### motivation  && background

1. 现有的大模型的长文本能力很差
2. 已经有一些研究增强大模型的长文本能力
3. 现有长文本基准测试不够完备

### Points

1. 这篇论文介绍了一个名为LongBench的多任务双语基准测试，旨在评估大型语言模型（LLMs）在长文本理解方面的能力。LongBench包含21个数据集，涵盖6个任务类别，包括单文档问答、多文档问答、摘要、少样本学习、合成任务和代码补全。这些任务覆盖了中英文，平均文本长度分别为英文6711个单词和中文13386个字符。所有数据集都标准化为统一格式，以便于自动评估LLMs。
2. 在主流大模型上进行了测试
3. 探讨了 rag 和 压缩技术对性能的影响
4. 对现有增强上下文的研究进行了探讨

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

## ACL2024: PreFLMR: Scaling up Fine-Grained Late-interaction Multi-modal Retrievers

### motivation && background

1. 现有的大模型对多模态的文本问答的效果很差
2. 缺少对于多模态检索器的 scaling up 的研究

### Points

1. 整合9个不同任务类型的数据集，M2KR
2. 在FLMR基础上，在不同阶段的训练训练检索器的组件
3. 使用不同参数的组件训练和测试,探究不同组件的参数量对结果的影响

## NeurIPS2024: TableRAG: Million-Token Table Understanding with Language Models

### motivation && background

1. 现在的表格理解需要将表格数据和任务一起输入，导致输入太长
2. 太长的文本会导致 loss-in-middle 的问题
3. 太长的表格数据需要消耗大量token,限制了其他token的长度

### Points

1. 现有的方法只看Schema,或者部分列和其数据，或者直接构造子表
2. TableRAG检索Schema和检索cell融合，只提取关键信息
3. 构造了两个数据集，TableRAG在上面表现出sota的效果
4. 合成了不同大小的表格数据，用来测试表格问答，检索性能和实际性能

# 模型轻量化

## ACL2024: Light-PEFT

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

## ACL2024: Pruning LargeLanguage Models to Intra-module Low-rank Architecture with Transitional Activations

waitting to update

# 幻觉

## 2023.11: Woodpecker: Hallucination Correction for Multimodal Large Language Models

### motivation && background

1. 多模态大模型的幻觉问题，此前的研究都在做insturction fine-tuning 都需要通过训练的方式
2. 之前的研究聚焦于事实性的错误

### Points

1. 提出一个无需训练的框架 Woodpecker, 通过5个步骤，最后产生答案
2. 聚焦于视觉的幻觉，而不是事实性的错误
3. 每一步骤都使用特定的模板或者特定的模型来生成
4. 在主流大模型的测试上都有较多提升

## Insight into LLM Long-context Failures: When Transformers know but don't tell

### motivation && background

1. 大模型不能完全利用长文本输入的全部信息
2. 通常只关注头部和尾部的信息
3. 长输入的中间部分被忽略
4. 现有方法尝试解决这些偏差

### Abstract

本文尝试从模型的中间激活值来评定长文本不同部位的信息在 模型的各个层中的损失，从而论证大模型对长文本的输入是有感知，但是输出却损失了信息

### conclusion

1. 中间的输入需要更多层才能感知到
2. 越前面的输入对应的精度越高

## CVPR2024: OPERA: Allevaiting Hallucination in Muti-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation

### background

1. 现有减轻大模型幻觉的方法是通过额外训练或者引入外部数据
2. 幻觉产生的原因：注意力权重。eg. 多模态的模型聚焦于总结性的token, 而忽略了image token ，导致输出出现幻觉

### Abstract

本文提出一种无需额外训练或者引入外部知识为代价，仅在在解码过程中解决幻觉的方法。主要是基于beam serach的方法，每次对候选token的概率分数施加惩罚项，并在一定窗口大小内判断是否需要回溯到summary token重新选择candiate。

### Conclusion

1. OPERA 不需要任何额外的训练和外部数据开销
2. 在四个多模态模型上表现不俗

### defect

1. OPERA 在短的回答上效果比较弱，在长回答中能发挥比较好的效果,可以通过调整参数和提升算法敏感度来解决。（可以做的比较少了）
2. OPERA 并不能解决所有幻觉现象，这取决于模型本身的能力，这是训练和数据要解决的问题

# 多模态

## ACL2024: Mutilmodal Table Understanding

### background

1. 现有的表格理解基本上都是以文本的形式(md,html)的形式输入给模型进一步完成任务
2. 现有的基准测试不够完善
3. 基于以上，现在的模型和测试都不能完全体现模型的表格理解能力

### Points

1. 统一表格图像来作为表格输入，形式统一，并且能够增强模型的表格理解
2. 构造了benchmark MMTab 来比较全面的评估了多模态表格理解的能力
3. 基于LlaVa 训练了 Table-LlaVa 在各项基准上为sota

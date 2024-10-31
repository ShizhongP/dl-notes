# UniIR: Training and Benchmarking Universal Multimodal Information Retrievers

- [论文连接](https://arxiv.org/pdf/2311.17136)
- [代码连接](https://github.com/TIGER-AI-Lab/UniIR?tab=readme-ov-file)
- 滑铁卢大学
- arxiv 2023.11.28
- ECCV 2024

## Kimi总结

这篇论文介绍了UniIR，这是一个多模态信息检索系统，旨在通过指令调用来处理和执行多种跨模态检索任务。UniIR能够理解和执行用户的指令，从包含数百万候选的异构池中检索出与查询相关的信息。以下是该论文的主要内容总结：

### 标题与作者

- **标题**: UniIR: Training and Benchmarking Universal Multimodal Information Retrievers
- **作者**: Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, Wenhu Chen
- **机构**: University of Waterloo, Georgia Institute of Technology, Hong Kong University of Science and Technology, Google DeepMind

### 摘要

- UniIR是一个统一的、指令引导的多模态检索器，能够处理八种不同的跨模态检索任务。
- 该系统在十个不同的多模态IR数据集上进行联合训练，能够根据用户指令执行各种检索任务，并展示出对新任务的零样本泛化能力。
- 论文还构建了M-BEIR，一个多模态检索基准，包含十个数据集，涵盖八种不同的多模态检索任务。

### 引言

- 信息检索（IR）是一项关键任务，涉及从大量数据中检索相关信息以满足特定用户需求。
- 现有的多模态IR研究通常关注单一领域的同质检索场景，不足以满足用户的多样化信息需求。

### UniIR框架

- UniIR框架旨在构建一个能够执行任何检索任务的单一检索器。
- 该框架通过指令调用来训练，能够解释用户的指令并执行各种检索任务。

### M-BEIR基准

- M-BEIR是一个大规模的多模态检索基准，包含来自多个领域的十个数据集，涵盖八种不同的多模态检索任务。
- 每个任务都伴随着人类编写的指令，总共包含150万查询和560万检索候选。

### 实验

- 作者评估了多种多模态检索模型在M-BEIR数据集上的性能，包括预训练的视觉-语言变换器模型。
- 实验结果表明，UniIR模型在零样本设置下能够很好地泛化到未见任务和数据集。

### 结论

- UniIR框架使得构建一个统一的多模态信息检索器成为可能，该检索器能够根据自然语言指令执行多样化的信息检索任务。
- M-BEIR基准的构建为未来多模态检索模型的研究提供了基础。
- 尽管现有模型性能仍有提升空间，但大规模预训练算法和更强的视觉-语言骨干模型有望为未来的发展奠定基础。

这篇论文的核心贡献在于提出了一个能够处理多种跨模态检索任务的统一框架UniIR，并通过构建M-BEIR基准来评估和训练这些模型，展示了在多任务学习和指令调优方面的优越性能。

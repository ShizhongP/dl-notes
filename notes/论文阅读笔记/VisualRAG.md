# VisualRAG

- [论文连接](https://arxiv.org/abs/2410.10594)
- [代码仓库](https://github.com/OpenBMB/VisRAG)
- 面壁智能与清华大学
- 2024.10.14
- Work in progress, arxiv, 现未有会议和期刊接收

## Kimi总结

这篇论文介绍了一个名为VisRAG（Vision-based Retrieval-augmented Generation）的系统，它是一个基于视觉的语言模型（VLM）的检索增强生成（RAG）管道。VisRAG旨在解决现有RAG系统仅基于文本，无法利用布局和图像等视觉信息的问题，这些视觉信息在现实世界中的多模态文档中扮演着重要角色。

**主要贡献和特点包括：**

1. **直接嵌入文档图像**：VisRAG不首先解析文档以获取文本，而是直接使用VLM将文档作为图像嵌入，然后检索以增强VLM的生成能力。

2. **保留原始文档信息**：与基于文本的传统RAG相比，VisRAG最大化了原始文档中数据信息的保留和利用，消除了解析过程中引入的信息丢失。

3. **训练和评估数据集**：作者收集了开源和合成数据来训练VisRAG中的检索器，并探索了多种生成方法。

4. **性能提升**：实验表明，VisRAG在检索和生成阶段都优于传统的基于文本的RAG，实现了比传统基于文本的RAG管道高25-39%的端到端性能提升。

5. **有效的数据利用和泛化能力**：进一步分析表明，VisRAG在利用训练数据和展示强大的泛化能力方面是有效的，使其成为处理多模态文档的RAG的有前途的解决方案。

6. **代码和数据的可用性**：作者提供了VisRAG的代码和数据，以便社区进一步研究和应用。

**实验结果**：

- 在多模态文档的检索方面，VisRAG-Ret（VisRAG的检索组件）展现了优越的性能，超过了现有的基于文本和视觉的检索器。
- 在生成方面，VisRAG-Gen（VisRAG的生成组件）超越了传统的基于文本的生成器。
- 使用GPT-4o（能够处理多个图像的VLM）时，VisRAG显示出随着检索到的文档数量增加，性能提升的潜力，表明了未来在多页推理方面的潜力。

**结论**：
VisRAG通过使用VLMs来促进RAG管道中的检索和生成，消除了传统基于文本的RAG所需的解析阶段。实验结果表明，VisRAG在检索和生成方面一致优于基于文本的RAG，同时保持了更简单的管道结构。作者希望VisRAG能激发未来RAG的发展，以纳入VLMs来处理多模态文档。

## 阅读笔记

动机:现有的RAG基本上是基于文本的，我们希望将文档图像和文本直接嵌入，做一个多模态的检索。通过检索得到的documents，交给多模态大模型去生成答案

方法论：

1. 使用多模态模型对文档图像和文本进行编码
2. 探究了不同模型(OCR,Caption),使用不同数据(in-domain,out-of-domain)情况下对retriver效果的影响
3. 对于只能输入一张图像的多模态大模型，使用文档图像拼接和根据权重选择的办法，来选择最可靠的答案
4. 对于能接受多张图像的多模态大模型(Minicpm-v,Qwen-v,GPT-4o)，直接将检索到的图像输入进去得到答案

结论:

1. 对于多张图像的输入的处理仍然是个问题，主要是基座模型的架构的设计和训练
2. 利用视觉RAG比起单纯的文本RAG效果更好

## 相关论文

- [Uniir](https://arxiv.org/abs/2311.17136)
- [Unirag](https://arxiv.org/abs/2405.10311)
- [MARVEL](https://arxiv.org/abs/2310.14037)

- [Unifying Multimodal Retrieval via Document Screenshot Embedding](https://arxiv.org/abs/2406.11251)
- [Unified Embedding for Multimodal Retrieval via Frozen LLM](https://aclanthology.org/2024.findings-eacl.105/)
- [ColPali](https://arxiv.org/abs/2407.01449)

# 一.计算机基础

[csapp 计算机基础课](https://hansimov.gitbook.io/csapp)

## Python 基础

[easy-py 编程入门书](https://github.com/FZU-psz/easy-py)

正在更新下一部分的内容，基本的深度学习知识和框架

## 计算机网络

OSI 5层，TCP可靠传输，VPN, port , socket...

## 操作系统

进程，线程，死锁，调度算法，并发控制，地址转换，页表，cache...

# 二.数学基础

## 高数

微分，偏导，链式法则....等相关定义和计算

## 概率论

贝叶斯公式,大数定理,大数定理,中心极限定理,最大似然估计

## 线性代数

- [3B1b 视频讲解，如何理解线性代数里面的操作](https://www.bilibili.com/video/av6731067/?vd_source=ec674d7bf8a6cdd072b8017efe791d9f)

- [3B1b博客笔记](https://charlesliuyx.github.io/2017/10/06/%E3%80%90%E7%9B%B4%E8%A7%82%E8%AF%A6%E8%A7%A3%E3%80%91%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%E7%9A%84%E6%9C%AC%E8%B4%A8/)

- [顶级的线性代数的理解](https://www.zhihu.com/question/20534668)

# 三.AI

## 传统机器学习

KNN,Kmeans,PCA,SVM,Logistic regression，梯度下降法, [beam search](https://zhuanlan.zhihu.com/p/82829880)

## 深度学习

- [PyTorch入门-小土堆](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=ec674d7bf8a6cdd072b8017efe791d9f)

- [跟李沐学AI](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)

- [《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/)

## 卷积神经网络CNN

这部分看李沐的视频够了

## 图神经网络GNN

参考博客:

- <https://zhuanlan.zhihu.com/p/75307407>

- <https://github.com/SivilTaram/Graph-Neural-Network-Note>

GNN 思想：**利用图的节点信息去生成节点（图）的 Embedding 表示**。

## 循环神经网络RNN

[RNN and LSTM](./notes/RNN.md)

## Transformer

**参考博客**

- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)
- [Attention注意力机制与self-attention自注意力机制](https://zhuanlan.zhihu.com/p/265108616)
- [注意力机制综述](https://zhuanlan.zhihu.com/p/631398525)
- [张俊林讲解attention](https://zhuanlan.zhihu.com/p/37601161)
- [kv cache](https://zhuanlan.zhihu.com/p/630832593)
- [为什么用kv cache 不 cache q?](https://www.zhihu.com/question/653658936/answer/3545520807)

- [B站视频讲解:王木头学科学]()

**源码实现**
[Transformer src](https://github.com/FZU-psz/dl-notes/tree/main/src/model/transformer)

## Bert

- [Bert 讲解](https://zhuanlan.zhihu.com/p/103226488)

- [bert 笔记](./notes/Bert.md)

## Llama2

- [知乎llama2结构详解](https://zhuanlan.zhihu.com/p/649756898)

- [llama note 笔记](./notes/Llama2.md)

## Deep Seek

Moe的结构

- [苏剑林讲MLA](https://kexue.fm/archives/10091)
- [deepseek-v2 笔记](./notes/DeepSeek-V2.md)

## 大模型训练

### 三大并行手段

[数据并行，张量并行，流水线并行](./notes/三大并行手段.md)

### Megatron-LM

- 参考博客:[知乎博客源码讲解](https://zhuanlan.zhihu.com/p/366906920)

- 视频讲解：[Megatron-LM讲解-李沐](https://www.bilibili.com/video/BV1nB4y1R7Yz/?spm_id_from=333.999.0.0)

### Deepspeed

- 参考博客 [deespeed 讲解](https://zhuanlan.zhihu.com/p/513571706),[deepspeed 讲解](https://blog.csdn.net/v_JULY_v/article/details/132462452)

- 文档: [deepspeed.ai](https://www.deepspeed.ai/getting-started/)

- 视频讲解：[deepspeed zero 讲解-李沐](https://www.bilibili.com/video/BV1tY411g7ZT/?spm_id_from=333.999.0.0)

### Flashattention

Flashattention 现在已经算是必不可少的算子了，基本上主流框架都实现了

## 大模型推理部署

论文推荐阅读：[A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)

- [vLLM](https://github.com/vllm-project/vllm)

- [Light-llm](https://github.com/ModelTC/lightllm)

- [TensorRT 半开源](https://github.com/NVIDIA/TensorRT-LLM)

- [SGLang](https://github.com/sgl-project/sglang)

## 大模型微调

全参数微调  与 高效参数微调

- 全参数微调: 将预训练模型作为初始化权重，对全部参数都进行更新
- 高效参数微调: 通常指对部分参数进行更新

### Prefix-tuning

waitting to update

### Lora

[Lora note](./notes/Lora.md)

### QLoRA

区别于LoRA,是在训练时进行量化,大致了解

## 量化技术

### GPTQ

论文：[GPTQ](https://arxiv.org/abs/2210.17323)

算法实现:[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)

### AWQ

论文: [AWQ](https://arxiv.org/abs/2306.00978)

算法实现:[llm-awq](https://github.com/mit-han-lab/llm-awq)

## 上下文缓存技术

- kv cache过长
- Prefill 阶段计算密集
- Decode 访存密集

[大模型推理仓库合集](https://github.com/DefTruth/Awesome-LLM-Inference?tab=readme-ov-file)

## RAG(Retrieval-Augmented Generation)

RAG综述：[Retrieval-Augmented Generations for Large Language Model: A Survey](https://arxiv.org/abs/2312.10997)

代码仓库

1. [LangChain](https://python.langchain.com/v0.1/docs/get_started/quickstart/)
2. [LlamaIndex](https://pypi.org/project/llama-index/)

## 多模态

多模态大模型综述: [A survey on Multimodal Large Language Models](https://arxiv.org/abs/2306.13549)

- [Clip](https://zhuanlan.zhihu.com/p/521151393)
- [Blip](https://zhuanlan.zhihu.com/p/640887802)
- [Flamingo](https://blog.csdn.net/LoseInVain/article/details/136072993)
- [LlaVa](https://blog.csdn.net/qq_58400270/article/details/135073408)

现有的多模态大模型的基本架构

1. 其他模态的数据的编码(图像，音频等)
2. 不同不模态之间的数据的适配器，或者叫connector
3. 基础的LLM,是直接采用还是微调对齐之后，或者从零开始训练

可以继续探究的点

1. 长文本和多模态融合
2. 继续挖掘多模态大模型的的能力，多模态的ICL,Cot
3. 具身智能，比较交叉了
4. 多模态生成的安全问题,主要还是对齐
5. 多模态大模型的更多的模态的对话能力

## 阅读推荐

[深入理解pytorch机制](https://www.cnblogs.com/rossiXYZ/p/15518457.html)

[llm-action 大模型实战和技术路线](https://github.com/liguodongiot/llm-action?tab=readme-ov-file)

[micrograd 小型深度学习框架，仅用200行不到的代码!](https://github.com/karpathy/micrograd)

## 论文阅读推荐

[papers](https://github.com/FZU-psz/dl-notes/tree/main/papers)

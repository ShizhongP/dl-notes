# 一.计算机基础

[csapp 计算机基础课](https://hansimov.gitbook.io/csapp)

## Python 基础
[easy-py 编程入门书](https://github.com/FZU-psz/easy-py)

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

KNN,Kmeans,PCA,SVM,Logistic regression，梯度下降法

## 深度学习

- [PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=ec674d7bf8a6cdd072b8017efe791d9f) b站小土堆

- [跟李沐学AI](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497) b站李沐

- [《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/)

## 卷积神经网络CNN

这部分看李沐的视频够了

## 图神经网络GNN

参考博客:

- <https://zhuanlan.zhihu.com/p/75307407>

- <https://github.com/SivilTaram/Graph-Neural-Network-Note>

GNN 就是做了这么一件事情：**利用图的节点信息去生成节点（图）的 Embedding 表示**。就是那么一个 Embedding 的方法。

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

## Llama2

- [知乎llama2结构详解](https://zhuanlan.zhihu.com/p/649756898)

- [llama note 笔记](./notes/Llama2.md)

## 大模型训练

参考博客：

- <https://zhuanlan.zhihu.com/p/688873027>

### 传统并行手段

**了解下torch的通信原语**

- [知乎教程](https://zhuanlan.zhihu.com/p/478953028)
- [pytorch文档教程](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training)

根据教程完成p2p通信和collective communication

#### 数据并行

参考:

- [DP与DDP原理解读](https://blog.csdn.net/ytusdc/article/details/122091284)

- [原理简单解读和DDP详细使用教程](https://github.com/KaiiZhang/DDP-Tutorial/blob/main/DDP-Tutorial.md)

**torch.nn.DP(Data Parallel)**

- DP是单进程多线程的形式,torch源码，受python的GIL的限制，至于什么是GIL,回顾python的基础知识

**DDP(Data Distributed Parallel)**

- 多进程的形式，一般一张显卡对应一个进程,通过多进程，绕过了GIL,性能较多线程可能更好

#### 流水线并行

视频讲解:<https://www.bilibili.com/video/BV1v34y1E7zu/?spm_id_from=333.999.0.0>

#### 张量并行

在llama的note讲解过了

### Megatron-LM

参考博客:<https://zhuanlan.zhihu.com/p/366906920>

视频讲解：<https://www.bilibili.com/video/BV1nB4y1R7Yz/?spm_id_from=333.999.0.0>

### Deepspeed

区别于其他框架的最大特点是Zero

文档:<https://www.deepspeed.ai/getting-started/>

视频讲解：<https://www.bilibili.com/video/BV1tY411g7ZT/?spm_id_from=333.999.0.0>

### Flashattention

理解原理！现在很多算法已经集成了Flashattention了，大部分不需要自己实现。

## 大模型推理部署

论文推荐阅读：[A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)

- [vLLM](https://github.com/vllm-project/vllm)

- [Light-llm](https://github.com/ModelTC/lightllm)

- [TensorRT](https://github.com/NVIDIA/TensorRT-LLM)

- [SGLang](https://github.com/sgl-project/sglang)

## 大模型微调

全参数微调  与 高效参数微调

- 全参数微调: 将预训练模型作为初始化权重，对全部参数都进行更新
- 高效参数微调: 通常指对部分参数进行更新

### Lora

[lora note](./notes/Lora.md)

### QLoRA

区别于LoRA,是在训练时进行量化,原理也是可以大致了解

## 量化技术

### GPTQ

论文：<https://arxiv.org/abs/2210.17323>

代码仓库: <https://github.com/IST-DASLab/gptq>

[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ),这个仓库集成了更多功能，支持很多模型

### AWQ

原理大致了解即可

论文:<https://arxiv.org/abs/2306.00978>

代码仓库:<https://github.com/mit-han-lab/llm-awq>

## 上下文缓存技术

如何解决 kv cache过长的问题!

<https://arxiv.org/abs/2406.17565>

## 阅读推荐

[自动微分](https://liebing.org.cn/automatic-differentiation.html#Reference)
[深入理解pytorch机制](https://www.cnblogs.com/rossiXYZ/p/15518457.html)

[llm-action 大模型实战和技术路线](https://github.com/liguodongiot/llm-action?tab=readme-ov-file)

[micrograd 小型深度学习框架，仅用200行不到的代码!](https://github.com/karpathy/micrograd)

## 论文阅读推荐

[papers](https://github.com/FZU-psz/dl-notes/tree/main/papers)

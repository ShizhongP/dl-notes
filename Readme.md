# 深度学习入门(更新中)

## 基础知识和能力

**前置知识:**

- 线性代数
- 微积分
- 概率论
- python基础语法(包含面向对象的知识)
- 深度学习框架pytorch的基本api调用

**基本的学习资料:**

- [PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=ec674d7bf8a6cdd072b8017efe791d9f) b站小土堆
- [跟李沐学AI](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497) b 站李沐

- [《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/)

## 后续学习

### 矩阵乘法

- [mm ref ](https://bhosmer.github.io/mm/ref.html)

###  ResNet残差神经网络学习

- [详解残差网络 ](https://zhuanlan.zhihu.com/p/42706477)

###  SGD和Adam优化器




###   Transformer结构

**墙裂推荐**

- [The Illustrated Transformer – Jay Alammar ](https://jalammar.github.io/illustrated-transformer/)
- [Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)
- [如何理解transformer各个层的意义](https://blog.csdn.net/rongsenmeng2835/article/details/110511294)

**attnetion 机制的理解和实现**

- [Attention注意力机制与self-attention自注意力机制](https://zhuanlan.zhihu.com/p/265108616)

- [注意力机制综述](https://zhuanlan.zhihu.com/p/631398525)

- [张俊林讲解attention](https://zhuanlan.zhihu.com/p/37601161) 这个文章有点老，但是讲的深入浅出，强烈推荐

- [pytorch 实现多头注意力](https://blog.csdn.net/weixin_53598445/article/details/125009686)

**Question**

- 什么是embding层?

- 以点积注意力机制为例，说明Q,K,V如何计算?

- 多头注意力机制中，是如何处理各个注意力机制的计算结果的？

- 机器翻译实例：在transformer架构中，给定一个句子"I have a cat"。阐述transformer是如何将其翻译为"我有一只猫"





## 加速训练，降低训练时显存的框架和技术

###  Gpipe

- [流水线并行,以Gpipe为例 - ](https://zhuanlan.zhihu.com/p/613196255)
- [GPipe论文精读【论文精读】_](https://www.bilibili.com/video/BV1v34y1E7zu/?spm_id_from=333.999.0.0&vd_source=ec674d7bf8a6cdd072b8017efe791d9f)

**Question**

- 什么是mini-batch?什么是micro-batch?
- Gpipe的性能评价(bubble)

###  DP or DDP

- [PyTorch 源码解读之 DP & DDP：模型并行和分布式训练解析](https://zhuanlan.zhihu.com/p/343951042)
- [数据并行上篇(DP, DDP) ](https://zhuanlan.zhihu.com/p/617133971)

**Question**

- 为什么要数据并行？什么是数据并行，具体举一个例子？DP瓶颈在哪里？

- 什么是异步梯度更新？

- 分布式数据并行(DDP)和数据并行区别是什么?

- Ring-AllReduce是什么，具体举个例子?

**task** 

- 查看pytorch提供的DP的源码



###  Zero(Deepspeed)

- [数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)
- [DeepSpeed之ZeRO系列：将显存优化进行到底 ](https://zhuanlan.zhihu.com/p/513571706)
- [Zero 论文精读【论文精读】](https://www.bilibili.com/video/BV1tY411g7ZT/?spm_id_from=333.788&vd_source=ec674d7bf8a6cdd072b8017efe791d9f)

**Question**

- Zero提出的目的是什么?
- zero-1,zero-2,zero-3分别干了什么事情?,在一次epoch中具体是如何计算的?
- zero-R具体干了什么事情

- 什么是zero-offload?



### Megatron-LM

- [张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228)
- [Megatron LM 论文精读【论文精读】](https://www.bilibili.com/video/BV1nB4y1R7Yz/?spm_id_from=333.337.search-card.all.click&vd_source=ec674d7bf8a6cdd072b8017efe791d9f)

**Question**

- Megatron 对mlp层，self-attention, 以及embeding层时如何切割和计算的?
- Megatron 对比 DP (通讯量,存储开销)

**task**

- 阅读Megatron-LM 源码

##  Colossal Ai

**阅读手册并部署**

- [Colossal-AI (colossalai.org)](https://colossalai.org/zh-Hans/)

### Flash Attention

- [ FlashAttention 的速度优化原理是怎样的?](https://www.zhihu.com/question/611236756/answer/3132304304)
- FlashAttention V2 的优化

### Lora微调技术

- [大模型微调（finetune）方法总结-LoRA,Adapter,Prefix-tuning，P-tuning，Prompt-tuning ](https://zhuanlan.zhihu.com/p/636481171)

###  LOMO

- [LOMO：利用有限的资源对大型语言模型进行全参数微调 ](https://zhuanlan.zhihu.com/p/638463057)



## 学习完之后应该具备的能力

- 使用pytorch 搭建任意的神经网络，只要给出原理图，都应该能实现
- 搭建一个完整训练和预测的pipeline
- 能使用各种加速训练和推理的框架，配置环境



## 杂谈

作为一名计算机类的学生(当然对其他专业的学生也是一样)，首先要具备自学的能力，而自学能力不仅是能深入理解知识，快速上手，能力迁移，也包括解决问题的能力。自己也还在探索的过程，我希望自己能真真切切做到以下几点

- 遇到问题，如何解决?

可以根据自己具备的基本知识，定位问题的关键，代码都是程序，都有运行逻辑，那一步错了，根据报错，思考下相关原理，尝试自己修改代码or相关配置。向 gpt 之类的提问，大部分时候他会给你一些灵感。上搜索引擎

- 如何写出优雅的工程代码?

多看 github 上的优质工程代码，学习以下，什么文件改放什么地方，学习如何组织一个大项目的开发，一个文件内的代码如何组织，函数要放在什么位置，代码结构又要怎么组织才算优雅?

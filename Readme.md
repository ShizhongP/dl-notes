# 408

## 计算机网络

OSI 5层，TCP可靠传输，VPN, port , socket...

## 操作系统

进程，线程，调度算法，并发控制，地址转换，页表，cache...

# 数学基础

## 高数

### 微分与偏导

## 概率论复习

### 贝叶斯公式



### 大数定理

参考博客：https://zhuanlan.zhihu.com/p/259280292

大数定理讨论的是多个随机变量的平均$ \frac{1}{n}\sum_{i=1}^n X_i$的渐进性质

对于一系列随机变量 $ \{X_n\}$，设每个随机变量都有期望。由于随机变量之和 $\sum_{i=1}^n X_i$ 很有可能发散到无穷大，我们转而考虑随机变量的均值 $\overline{X_n} = \frac{1}{n} \sum_{i=1}^n X_i$和其期望 $\mathbb{E}[\overline{X_n}]$之间的距离。若 ${X_n}$满足一定条件，当n足够大时，这个距离会以非常大的概率接近0，这就是大数定律的主要思想。

- 定义: 对于任意$ \epsilon >0$, 若恒有$ lim_{n-> +\infty}P(|\overline{X_n}-\mathbb{E}(\overline{X_n})|<\epsilon) =1$, 则称随机变量序列$\{X_n\}$满足大数定理

### 中心极限定理

中心极限定理讨论的是独立随机变量和 $Y_n = \sum_{i=1}^n X_i$ 的极限分布

$Y_n$ 可以看成是很多微小的随机因素$X_1,X_2,...X_n$之和,n很大，我们关心在什么条件下面$Y_n$的极限分布是正态分布

- 独立同分布中心极限定理（林德伯格-列维中心极限定理)

  如果随机变量$X_1,X_2....X_n$相互独立，且分布相同，他们的数学期望$\mu$和方差$\sigma^2$一致,则随机变量

  $$ Y_n = \frac{\sum_{i=1}^nX_i - n\mu}{\sqrt{n}\sigma}$$,当n较大的时候 $Y_n \sim N(0,1)$ ,近似标准正态分布，即$ \sum_{i=1}^nX_i \sim N(n\mu,n\sigma^2)$

- 二项分布中心极限定理(棣莫弗-拉普拉斯中心极限定理)

  $X$是n次伯努利实验中事件A出现的次数,p是每次时间A发生的概率，即$X \sim B(n,p)$

  当n较大时 $X\sim N(np,np(1-p))$​

### 最大似然估计

参考博客: https://zhuanlan.zhihu.com/p/26614750

- 极大似然估计，通俗理解来说，**就是利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值！**

- 具体步骤:

1. 给据给定样本和总体分布类型构造似然函数(目的就是让已知的样本能真实反映总体的分布，也就是，似然函数必须要最大)
2. 对似然函数取对数，求导数，令导数=0,此时根据等式可以求出估计的参数

## 线性代数

视频

- [3B1b 视频讲解，如何理解线性代数里面的操作](https://www.bilibili.com/video/av6731067/?vd_source=ec674d7bf8a6cdd072b8017efe791d9f)

博客

- [3B1b博客笔记](https://charlesliuyx.github.io/2017/10/06/%E3%80%90%E7%9B%B4%E8%A7%82%E8%AF%A6%E8%A7%A3%E3%80%91%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%E7%9A%84%E6%9C%AC%E8%B4%A8/)

- [顶级的线性代数的理解](https://www.zhihu.com/question/20534668)

# AI

## Python 基础

### 基本语法和概念

参考博客：https://www.liujiangblog.com/course/python/78

### 基础的第三方库

matplotlib,numpy,sklearn等,自行STFW学习

###  进阶学习

#### threading

- 线程（请自行STFW深刻理解）

  线程是比进程更小的调度单位。具体来说，一个程序的运行状态称做进程，进程被译码称多个指令，由1个或者多个线程分别承担一部分指令的工作，这样指令与指令之间执行的时候，发生切换的是线程。

  此时cpu的轮转就不再局限于进程之间，而是可能在同一个进程的不同线程之间的切换，或者不同进程之间的线程的切换。

- 代码实例
```python
import threading 
import time
def run(id):
    print(f"Thread {id} is running")
    time.sleep(1)
    print(f"Thread {id} is done")

if __name__ == "__main__":
    
    threads = []
    # 创建进程组
    for i in range(5):
        t = threading.Thread(target=run, args=(i,))
        threads.append(t)
        t.start()
    # 进程阻塞，直到所有线程完成
    for t in threads:
        t.join()
    #所有进程结束，回到主进程，执行主进程的程序
    print("All threads are done!")
```

- 思考

  如果同一个进程的线程之间由需要共享的资源，如何实现，如何避免资源请求冲突（实现互斥锁）？请根据前面的博客或者STFW

#### multiprocess

- 进程

  每个进程都拥有一个GIL，这样子多个进程之间就不会受一个GIL的限制，可能并发性高

- 代码实例

```python
import time
import multiprocessing

def run(i):
    print(f"Process {i} is running")
    time.sleep(1)
    print(f"Process {i} is done")
    
if __name__ == "__main__":
    
    #创建进程组 
    for i in range(5):
        p = multiprocessing.Process(target=run, args=(i,))
        p.start()
    print("All processes are started")
    
    #等待所有进程结束
    for p in multiprocessing.active_children():
        p.join()
    
    print("All processes are done!")
```

- 思考

  同样的，如何实现多个进程之间的资源共享?(这个很重要)

#### async

- 协程

  又称作微线程，相比于线程，线程之间的切换是由**程序本身控制的**，省去了切换进程之间的开销

- 代码实例

```python
import time
import asyncio
async def task():
    print("Task is started")
    # 模拟耗时操作 比如读取磁盘的数据
    await asyncio.sleep(1)
    return "Task is done"
    
async def task2():
    print("Task2 is started")
    await asyncio.sleep(1)
    return "Task2 is done"

if __name__ == "__main__":
    # 创建事件循环
    loop = asyncio.get_event_loop()
    # 创建任务,封装到future中
    tasks = [asyncio.ensure_future(task()), asyncio.ensure_future(task2())]
    # 将任务加入事件循环
    loop.run_until_complete(asyncio.wait(tasks))
    
    #获取时间的结果
    for t in tasks:
        print(t.result())
    # 关闭事件循环
    loop.close()
```

- 思考以及任务

  协程适合用于那种场景呢？为什么（请从cpu轮转的角度分析），试着使用协程写一个小型爬虫爬取任意一个网站的图片吧

## 传统机器学习

### KNN

```python
import numpy as np
from collections import Counter

def knn(X_train, y_train, X_test, k):
    """
    Implements the k-Nearest Neighbors (kNN) algorithm.
    
    Parameters:
    X_train (numpy.ndarray): Training data features.
    y_train (numpy.ndarray): Training data labels.
    X_test (numpy.ndarray): Test data features.
    k (int): Number of nearest neighbors to consider.
    
    Returns:
    numpy.ndarray: Predicted labels for the test data.
    """
    distances = np.sqrt(((X_train - X_test[:, None])**2).sum(axis=2))
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_labels = y_train[nearest_indices]
    return np.array([Counter(labels).most_common(1)[0][0] for labels in nearest_labels])

X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[2, 3], [6, 7]])
y_pred = knn(X_train, y_train, X_test, k=3)
print(y_pred)  # Output: [0 1]
```

### Kmeans

### SVM

最大化类别之间的间隔

### PCA 与 LDA

pca：https://www.zhihu.com/question/41120789/answer/481966094

### 逻辑回归

### 梯度下降法证明

- 最优化问题,局部最小值不一定是全局最优解，通过添加一些随机噪声/扰动能够跳出局部最优解
- 训练误差指的是在训练集上的表现，泛化误差指的是在全部数据集上的表现，有时可以说是在测试集上的表现

#### 一维梯度下降

考虑一维函数的随机梯度下降，一个连续可微实值函数$f : \mathbb{R} \rightarrow \mathbb{R}$ 利用泰勒展开可以得到

$ f(x+ \epsilon)= f(x) + \epsilon f'(x)+ O(\epsilon^2)$

即在一阶近似中，$f(x+\epsilon)$可通过x出的函数值$f(x)$和一阶导数$f'(x)$近似得出。我们可以假定负梯度方向上移动的$\epsilon$会减少$f$。为了简单起见，我们选择固定的步长$\eta>0$,然后令$ \epsilon = -\eta f'(x)$，然后将其代入泰勒展开式可以得到

$f(x-\eta f'(x)) = f(x) -\eta f'^2(x)+O(\eta^2 f'^2(x))$

如果$f'(x) \ne 0$导数并没有消失，那么可以将上面的泰勒展开式继续展开，因为$\eta^2 f'^2(x)>0$ 。此外，我们也可以令$\eta$小到让高阶函数不那么相关，因此

$$ f(x-\eta f'(x)) \approx f(x)$$

这就意味着我们可以使用$ x \leftarrow x-\eta f'(x)$来迭代x。直到某个终止条件停止迭代

#### 多维梯度下降

和一维梯度下降类似的过程,考虑变量$\mathbf{x} = [x_1,x_2,...x_d]^T$ 的情况。即目标函数$f: \mathbb{R}^d \rightarrow \mathbb{R}$，将向量映射为标量，相应的他的梯度也是多元的，由d个偏导数组成的向量:

$ \nabla f= [\frac{\partial f(\mathbf(x))}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2},...\frac{\partial f(\mathbf(x))}{\partial x_d}]^T$​ 

梯度中的每个偏导数$ \partial f/ \partial x_i$代表了当输入了$x_i$时f在$\mathbf{x}$处的变化率。和单变量一样，考虑使用泰勒展开式来近似

$f(\mathbf{x}+\mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^T \nabla f(\mathbf{x}) + O( \lvert\lvert\epsilon^2 \rvert\rvert)$

通过$ \mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$​​​来迭代求解

## 深度学习

- [PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=ec674d7bf8a6cdd072b8017efe791d9f) b站小土堆

- [跟李沐学AI](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497) b站李沐

- [《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/)

## 卷积神经网络CNN

## 图神经网络GNN

参考博客: 

- https://zhuanlan.zhihu.com/p/75307407

- https://github.com/SivilTaram/Graph-Neural-Network-Note

GNN 就是做了这么一件事情：**利用图的节点信息去生成节点（图）的 Embedding 表示**。就是那么一个 Embedding 的方法。

## 循环神经网络RNN

### RNN

参考博客：

- https://zhuanlan.zhihu.com/p/32085405

- https://colah.github.io/posts/2015-08-Understanding-LSTMs/

RNN 中的单个神经元如下所示

<img src="https://pic4.zhimg.com/80/v2-f716c816d46792b867a6815c278f11cb_720w.webp" alt="img" style="zoom:50%;" />

**x**表示当前状态的输入, **h**表示接受的上一个节点的输入, **y**是当前状态的输出，**h'**是传递给下一个状态的输入

从上面的图片可以看到，**h'**的计算与当前状态**x**和上一节点的输入**h**有关，**y**的计算通常由**h'**计算得来

如干个这个样的单元组成一个序列即为**RNN(recurrent neural network)**循环神经网络,如下图所示

<img src="https://pic2.zhimg.com/80/v2-71652d6a1eee9def631c18ea5e3c7605_720w.webp" alt="img" style="zoom:50%;" />

### LSTM

长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

LSTM和RNN的输入的区别如下图

<img src="https://pic4.zhimg.com/80/v2-e4f9851cad426dfe4ab1c76209546827_720w.webp" alt="img" style="zoom:50%;" />

相比于RNN只有一个传递状态**h**, LSTM有两个传递状态$c^t和 h^t$, (lstm 的$h^t$应该对应的是rnn的$h^t$​)

通常$c^t$是上一个状态传来的$c^{t-1}$​加上某些数值

具体的计算结构如下

1. 遗忘阶段: 计算$f_t$,选择那些元素需要被遗忘
2. 记忆阶段: 计算$i_t和\tilde{C_t}$​,然后将两者按元素相乘，选择那些元素需要被记忆
3. 更新阶段:根据 $f_t与c_{t-1}和 i_t与 \tilde{C_t}$计算$c_t$
4. 输出阶段: $h_t$经过某些变化和$c_t$ 计算当前单元的输出



<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" alt="img" style="zoom: 25%;" /><img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" alt="img" style="zoom:25%;" />

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" alt="img" style="zoom:25%;" /><img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" alt="img" style="zoom:25%;" />

<img src="https://pic4.zhimg.com/v2-03c41f0aaee75d920c1ba7bd756ae207_r.png" alt="preview" style="zoom: 25%;" />





## Transformer

**参考博客**

- [The Illustrated Transformer – Jay Alammar ](https://jalammar.github.io/illustrated-transformer/)
- [Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)
- [Attention注意力机制与self-attention自注意力机制](https://zhuanlan.zhihu.com/p/265108616)
- [注意力机制综述](https://zhuanlan.zhihu.com/p/631398525)
- [张俊林讲解attention](https://zhuanlan.zhihu.com/p/37601161) 文章有点老，但是讲的很好
- [kv cache](https://zhuanlan.zhihu.com/p/630832593)
- [为什么用kv cache 不 cache q?](https://www.zhihu.com/question/653658936/answer/3545520807)



## Llama2 

参考博客

- [知乎llama2结构详解](https://zhuanlan.zhihu.com/p/649756898)

### 模型结构

Llama 2的模型结构与标准的Transformer Decoder结构基本一致，主要由32个 Transformer Block 组成，不同之处主要包括以下几点：

1. 前置的**RMSNorm**层
2. Q在与K相乘之前，先使用**RoPE**进行位置编码
3. **K V Cache**，并采用**Group Query Attention**
4. FeedForward层

那么下文将结合具体的代码来展开聊一聊这些差异

#### RMSNorm

Transformer的Normalization层使用的是LayerNormlization层归一化对Tensor进行归一化

**RMSNorm**是LayerNormlization的变体，它省去了求均值的过程，也没有偏置。

#### RoPE旋转位置编码

参考博客: 

- https://zhuanlan.zhihu.com/p/642884818
- https://zhuanlan.zhihu.com/p/647109286

旋转位置编码，使在计算attention的时候，也够考虑相对位置信息

可以将模型外推，在预测的时候，可以通过旋转矩阵来达到超过训练长度的预测，还有就是捕获一些相对位置的信息

####  GQA （分组查询注意力机制）

参考博客：[MHA, MQA, GQA](https://mp.weixin.qq.com/s/_4OxoRLxhOcjGf0Q4Tvp2Q)

###  实战

[llama-factory](https://github.com/hiyouga/LLaMA-Factory) 里面集成了各种技术

## 大模型预训练框架

参考博客：

- https://zhuanlan.zhihu.com/p/688873027

### 传统并行手段

**先了解torch的通信原语**

- [知乎教程](https://zhuanlan.zhihu.com/p/478953028)
- [pytorch文档教程](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training)

根据教程完成p2p通信和collective communication(独立完成，勿借助任何ai工具或者copy paste)

#### 数据并行

[DP与DDP原理解读](https://blog.csdn.net/ytusdc/article/details/122091284)

[原理简单解读和DDP详细使用教程](https://github.com/KaiiZhang/DDP-Tutorial/blob/main/DDP-Tutorial.md)

**DP(Data Parallel)**

- DP是单进程多线程的形式,可以去看torch源码，受python的GIL的限制，至于什么是GIL,请回顾python的基础知识或STFW

  ```python
  #init dataset
  train_loader = ...
  test_loader = ...
  #init model 
  model = ...
  optimizer = ...
  loss_fn = ...
  
  # DP
  model = nn.DataParallel(model)
  
  #train 
  train(model,optimizer,loss_fn)
  ```

**DDP(Data Distributed Parallel)**

- 多进程的形式，一般一张显卡对应一个进程

#### 流水线并行



#### 张量并行（模型并行）



### Megatron-LM

https://zhuanlan.zhihu.com/p/366906920

### Deepspeed 



### Flashattention

### Flashattention v2

### Flashattention v3

## 大模型推理部署技术

论文推荐：[A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)

### vLLM

### Light-llm

### TensorRT

## 大模型微调技术

**全参数微调  与 高效参数微调**

- 全参数微调: 将预训练模型作为初始化权重，对全部参数都进行更新
- 高效参数微调: 通常指对部分参数进行更新

**什么是微调?个人理解（瞎写）**

​	对于一个预训练好的模型，如果我们希望通过重新调整一些模型参数，使得这个新的模型在新的任务上能有更好的表现，是不是感觉是废话，因为我这讲的不就是微调吗:) 。

​	前面已经对微调进行大致的分类的对吧，如果在原来整个模型上继续训练的叫做全参数微调，但这样子的话，成本就和你做预训练不就一样了吗。这样子产生了一个问题，如果某个机构开源了一个大模型，整个模型是在4张A800(80GB)上完成训练的，如果其他人希望能使用这个开源模型做一些下游任务，如果仍采用全参数微调，也得需要4张A800啊，那如果我没有那么多卡(没钱啊)，我要怎么完成这个任务呢，我们希望的是只训练一部分参数（不要全部参数都拿去训练了），就能在这个下游任务上取到好的结果，这就叫做**高效参数微调**（其实我觉得叫它**部分参数微调**更合适）。

### 提示词微调

### 指令微调

指令微调是一种通过在由（指令，输出）对组成的数据集上进一步训练LLMs的过程。其中，指令代表模型的人类指令，输出代表遵循指令的期望输出。这个过程有助于弥合LLMs的下一个词预测目标与用户让LLMs遵循人类指令的目标之间的差距。

指令微调可以被视为有监督微调（Supervised Fine-Tuning，SFT）的一种特殊形式。但是，它们的目标依然有差别。**SFT是一种使用标记数据对预训练模型进行微调的过程，以便模型能够更好地执行特定任务。（**也就是说只有带标签的数据输入？**）**而指令微调是一种通过在包括（指令，输出）对的数据集上进一步训练大型语言模型（LLMs）的过程，以增强LLMs的能力和可控性。**指令微调的特殊之处在于其数据集的结构，即由人类指令和期望的输出组成的配对。**这种结构使得指令微调专注于让模型理解和遵循人类指令

### LoRA

论文地址:  https://arxiv.org/pdf/2106.09685

参考博客:  https://zhuanlan.zhihu.com/p/623543497

代码仓库: 

LoRA，全称 Low-Rank Adaptation, 低秩适配?（看名字就知道和矩阵的秩有关）

​	对于预训练模型的参数$H$，我们在其上面进行微调（参数的更新），假设参数的变化为$ \Delta H  $ ,  那么更新过后的模型可以表示为$ H +\Delta H$​ 

​	具体一点，对于模型内的某一层的矩阵$W$，我们假设预训练模型这一层的参数为$W_0$， 假设其变化的参数为$\Delta W$，那么这一层参数上的更新可以表示为$ W_0+ \Delta W$。其中$W \in \mathbb{R}^{d \times k}$, 则也有$W_0,\Delta W \in \mathbb{R}^{d \times k}$，（形状要一样的啊，要不然两个矩阵怎么相加）

​	进一步，$\Delta W$是不是可以表示成两个矩阵相乘的形式呢？(肯定可以啊)。我们假设$\Delta W = AB$，其中$A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k},r\ll min(d,r)$，那么对于那么这一层参数的更新就可以表示为$W + \Delta W = W+ AB$。$r$就是LoRA中的秩序了，通常$r=1,2,3,4,8$​都不是一个太大的值，所以这就叫低秩。到这里LoRA的主体架构就讲完了，剩下的部分看论文或者博客去理解了。

## 大模型轻量化技术

### 剪枝

#### 结构化剪枝

#### 非结构化剪枝

### 蒸馏

### 量化

#### GPTQ

论文：https://arxiv.org/abs/2210.17323

实战仓库: 

#### AWQ

## 上下文缓存技术

目前这方面的文献还不是很多，开源的技术不是太多

https://arxiv.org/abs/2406.17565

## 论文推荐



## 仓库推荐

[llm-action 大模型实战和技术路线](https://github.com/liguodongiot/llm-action?tab=readme-ov-file)


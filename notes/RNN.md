### RNN

参考博客：

- <https://zhuanlan.zhihu.com/p/32085405>

- <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>

RNN 中的单个神经元如下所示

<img src="https://pic4.zhimg.com/80/v2-f716c816d46792b867a6815c278f11cb_720w.webp" alt="img" style="zoom:50%;" />

**x** 表示当前状态的输入, **h** 表示接受的上一个节点的输入, **y** 是当前状态的输出，**h'** 是传递给下一个状态的输入

从上面的图片可以看到，**h'** 的计算与当前状态 **x** 和上一节点的输入 **h** 有关，**y** 的计算通常由 **h'** 计算得来

如干个这个样的单元组成一个序列即为 **RNN(recurrent neural network)** 循环神经网络,如下图所示

<img src="https://pic2.zhimg.com/80/v2-71652d6a1eee9def631c18ea5e3c7605_720w.webp" alt="img" style="zoom:50%;" />

### LSTM

长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

LSTM和RNN的输入的区别如下图

<img src="https://pic4.zhimg.com/80/v2-e4f9851cad426dfe4ab1c76209546827_720w.webp" alt="img" style="zoom:50%;" />

相比于RNN只有一个传递状态**h**, LSTM有两个传递状态 $c^t和 h^t$ , (lstm 的 $h^t$ 应该对应的是rnn的 $h^t$ ​)

通常 $c^t$ 是上一个状态传来的 $c^{t-1}$ ​加上某些数值

具体的计算结构如下

1. 遗忘阶段: 计算 $f_t$,选择那些元素需要被遗忘
2. 记忆阶段: 计算 $i_t和\tilde{C_t}$ ​,然后将两者按元素相乘，选择那些元素需要被记忆
3. 更新阶段:根据 $f_t与c_{t-1}和 i_t与 \tilde{C_t}$计算 $c_t$
4. 输出阶段: $h_t$经过某些变化和$c_t$ 计算当前单元的输出
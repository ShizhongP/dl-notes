
reference: [bert](https://zhuanlan.zhihu.com/p/103226488)

## Bert

### 结构

![bert](https://pic1.zhimg.com/80/v2-d0a896547178320eb21a92550c48c66a_720w.webp)

前面已经学习过了Transformer,而Bert主要是由Transformer中的encoder 模块堆叠而成

#### 输入层

1. token embedding ：将输入进行一层embedding表示输入的语义
2. Segmet embedding: 为了区别输入的两个句子，对不同句子的token的位置上分别编码0和1
3. Position embedding: 不同于transformer,bert的位置编码的embedding 是可以学习的参数。原因：bert的数据量足够大，能够学习到位置信息。缺陷：不能超过训练的位置编码的长度,如训练时是512,输入不能超过512

#### 预训练的损失函数

交叉熵，本质上都是分类任务

#### cls 的作用

cls 这个无明显语义特征的信息能更加公平的融合文本中的各个token的信息

#### 如何解决512长度的限制

1. 分段，每个短输入bert使用，最大池化层，平均池化层，
2. 头截断
3. 尾阶段
4. 截取部分头，截取部分尾,拼接
5. [层次分解位置编码- 苏剑林](https://spaces.ac.cn/archives/7947)

### 使用场景

#### Masked Language Model (完形填空)

1. 动机：让模型学会预测词，做完形填空

2. 方法论： Mask总体的 15% 的token：对于80%的token 替换成 \<mask\>,对于10% 的token 替换成已有的token , 对于剩下10% 保持不变

#### Next Sentence Prediction

1. 动机：适配下游任务，需要预训练模型学习句子之间的关系

2. 方法论：50% 的数据中的句子来自于同一个文档中的上下文，50% 的数据中的句子不是来自同一个文档的上下文

### 下游任务(fine tuning)

1. 句子相似度推测
2. QA问答
3. 推理任务
4. 文本分类和序列标注

### 参数量

bert base 110M
bert large 340M

### Bert 变种

#### RoBERTa

1. 更大的batch_size, seq_len
2. 移除了next sentence prediction ，实验表明这样效果更好
3. 添加了sentence order prediction, 就是预测两个句子之间有没有交换过顺序
4. 静态mask 变成 动态mask,不同训练的mask的位置不同

#### ALBERT (A Lite Bert)

1. Embedding 拆成两个小矩阵相乘
2. 只有一个transformer block, 像gpt一样，仿佛通过改层block ，通过若干次得到最后输出


## Prefix-Tuning

受 prompt 的启发，针对不同的任务，我们通常都有一段 task description + input 。但是这样的话，针对不同的任务，我们都需要构造不同的 task description ， 而 Prefix-Tuning 希望通过一组可学习参数来表征 task description

具体的做法就是在 LLM 旁路增加一组 visual token embedding 用来学习和更新，而原始的参数冻结,在自回归生成中，前 n 个 output 使用该旁路分支，直到真正的输入进来，即 n+1 个输出使用LLM。
$ h_i = \begin{cases} P[i:] & \text{if i<n} \\ LLM(z_i,h_{<i})  & \text{else} \end{cases} $

![Prefix-Tuning](https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/transformer/PrefixTuning_exam.png)

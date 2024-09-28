
几个指标
1. TTFT(Time of the first token): 第一个token产生的时间
2. TBT(Time between token): 上一个token产生到下一个token的时间

prefill 是计算密集型,计算原始的kv cache
decode 是访存密集，需要频繁的使用kv cache
batch(to utilize full capability of machine), cache(to reuse), shcedule(to improve efficiency ),三个核心要点

## Architecture

1. client

2. server
    - tokenization
    - engine (mutiple gpu) 
    - detokenization
每个节点有一个contorller负责数据并行和调度


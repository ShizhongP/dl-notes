
batch(to utilize full capability of machine), cache(to reuse), shcedule(to improve efficiency ),三个核心要点

## Architecture

1. client

2. server
    - tokenization
    - engine (mutiple gpu) 
    - detokenization
每个节点有一个contorller负责数据并行和调度


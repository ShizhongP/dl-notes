import torch.distributed as dist
import torch.multiprocessing as  mp
import torch
import os
'''
 使用torch 实现 scatter通信模式，其他通信原语言也是类似的
'''
def init_process(rank, size,backend='gloo'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    t = torch.tensor([rank])
    if rank == 0:
        scatter_list = [torch.tensor([i*i]) for i in range(size)]
        dist.scatter(tensor=t, src=0, scatter_list=scatter_list)
    else :
        print(f"Rank  {rank} : {t} ")
        dist.scatter(tensor=t, src=0)
        print(f"Rank {rank} received: {t} ")
    
if __name__ =="__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    word_size =  4
    mp.spawn(init_process, args=(word_size,), nprocs=word_size, join=True)
    
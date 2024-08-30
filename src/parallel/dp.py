import torch 
import torch.nn as nn
import threading
from typing import List
# from src.model.LeNet import LeNet

class DataParallel(nn.Module):
    def __init__(self, module:nn.Module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()
        self.module = module
        # the device ids of the gpus
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ids = device_ids if not device_ids  else list(torch.cuda.device_count())
        # the output device 
        self.output_device = output_device if not output_device else self.device_ids[0]
        # Operating in the dim default dim is 0
        self.dim = dim 
        self.training = True
        
        if len(device_ids) == 0:
            self.modules.to(torch.device(self.device_type,device_ids[0]))
        
    def forward(self, *inputs, **kwargs):
        # if the device_ids is none, then use the model in the current device
        if self.device_ids == None:
            return self.module(*inputs)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0])
        
        for module in self.modules:
            module.train(self.training)
        
        # NOTE:Satter Stage
        inputs = self.scatter(inputs, self.device_ids)
        
        # NOTE:Replicate Stage
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        
        # NOTE:Parallel Apply Stage,for each mocule execute the forward function
        outputs = self.parallel_apply(replicas, inputs)
        
        # NOTE:Gather Stage,gather the result in the ouput device
        return self.gather(outputs, self.output_device)
    
    def scatter (self, inputs, kwargs, device_ids:List[int]):
        from torch.nn.parallel._functions import Scatter
        if isinstance(inputs,torch.Tensor):
            inputs = Scatter.apply(device_ids,None,self.dim,inputs)
        else:
            raise ValueError("The inputs should be a tensor")
    def replicate(self, module:nn.Module, device_ids:List[int]):
        from torch.nn.parallel._functions import Broadcast
        
        num_replicas = len(device_ids)
        
        #Step 1: get the paramters from all modules
        params = list(module.parameters())
        param_indices = {param: idx for idx, param in enumerate(params)}
        
        #Step 2: Broadcast the parameters to all gpus
        outputs = Broadcast.apply(device_ids,*params)
        param_copies = [outputs[i:i + len(params)] for i in range(0,len(outputs),len(params))]
        
        #Step 3: update all the modules's paramters
        modules = list(module.modules())
        module_copies = [[]]*num_replicas
        module_indices = {}
        # initialize 
        for i, module in enumerate(modules):
            module_indices[module] = i
            for j in range(num_replicas):
                replica = module._replicate_for_data_parallel()
                module_copies[j].append(replica)
        
        # copy operation
        for i, module in enumerate(modules):
            # copy the module ans param
            module_layers = module._modules.items()
            params = module._parameters.items()
            for module_item,param_item in zip(module_layers,params):
                module_key, module_layer = module_item
                param_key,param = param_item
                
                module_idx = module_indices[module_layer]
                param_idx  = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, module_key, module_copies[j][module_idx])
                    setattr(replica, param_key,  param_copies[j][param_idx]  )
        return [module_copies[j][0] for j in range(num_replicas)]
    
    def parallel_apply(self, replicas, inputs):
        pass 
    def gather(self, outputs, output_device):
        pass 


def main():
    batch_size = 2
    train_data = [torch.randn(4, 4) for _ in range(4)]
    n_gpu = 2
    model = nn.Linear(4, 4)
    
    ## parallize the the model 
    model = DataParallel(model, device_ids=[0, 1])
    res  =  model(train_data[0])
    print(res)

if __name__ == '__main__':
    main()


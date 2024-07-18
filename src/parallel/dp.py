import torch 
import torch.nn as nn
import torch.nn.functional as F
import threading
from src.model.LeNet import LeNet
''' DP 的实现是单进程多线程的'''
def train(model,optimizer):
    model.train()
    for epoch in range(epochs):
        for batch_idx in range(100):
            data, target = torch.rand(64,1,28,28), torch.randint(0,10,(64,))
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

if __name__ == "__main__":
    
    
    torch.manual_seed(0)
    # 加载数据
    
    # 加载模型 
    model = LeNet()
    
    ######## DP ###################
    model = nn.DataParallel(model)#
    ###############################
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 1
    

    train(model,optimizer)
    




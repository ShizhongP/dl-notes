import torch 
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self,):
        self.conv1 = nn.Conv2d(1,10,kernal_size=5)
        self.conv2 = nn.Conv2d(10,20,kernal_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10) # 10个类别
    
    def forward(self,x):
       x = F.relu(F.max_pool2d(self.conv1(x),2))
       x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
       x = x.view(-1,320)
       x = F.relu(self.fc1(x))
       x = F.dropout(x,training=self.training)#根据是否正在训练选择要dropout
       x = self.fc2(x)
       
       x = F.log_softmax(x)
       return x 
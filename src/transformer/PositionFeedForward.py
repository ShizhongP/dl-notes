import torch
import torch.nn as nn


class PositionFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=None):
        super(PositionFeedForward, self).__init__()
        
        self.linear_1 = nn.Linear(dim_model,dim_ff)
        self.relu= nn.ReLU()
        self.linear_2 = nn.Linear(dim_ff,dim_model)
        
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,x):
        # x: batch_size,sequence_length,dim_model
        
        output = self.linear_1(x) # batch_size,sequence_length,dim_ff
        output = self.relu(output) 
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.linear_2(output) # batch_size,sequence_length,dim_ff
        
        # simplify the process
        # output = self.linear_2(self.dropout(self.relu(self.linear_1(x))))
        
        return output
        
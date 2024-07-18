import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from src.model.LeNet import LeNet


def main():
    nn.DataParallel()
    nn.parallel.DistributedDataParallel()
    nn.parallel
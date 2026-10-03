import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Dropout(.2),
                               nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),
                               nn.Linear(64,num_classes))
    def forward(self,x): return self.net(x)

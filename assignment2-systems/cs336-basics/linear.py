import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self,in_features,out_features,device=None,dtype=None):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std = 2/(self.in_features+self.out_features)

        self.weight = nn.init.trunc_normal_(tensor=torch.empty(self.out_features,self.in_features),
        mean=0,
        std=std,
        a=-3*std,
        b=3*std)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        return torch.einsum("...i,ji->...j",x,self.weight)


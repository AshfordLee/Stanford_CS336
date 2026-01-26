import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):


        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.init.trunc_normal_(tensor=torch.empty(self.num_embeddings,self.embedding_dim),
        mean=0,
        std=1,
        a=-3,
        b=3)


    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
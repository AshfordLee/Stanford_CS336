import torch
import torch.nn as nn
import math
from einops import rearrange

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


class rmsnorm(nn.Module):

    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):

        super().__init__()

        self.d_model = d_model
        self.eps = eps 
        self.device = device
        self.dtype = dtype

        self.weights = nn.init.trunc_normal_(tensor=torch.ones(self.d_model))

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS_a = torch.sqrt(torch.einsum("...d,...d->...",x,x)/self.d_model + self.eps)

        return ((x/RMS_a.unsqueeze(-1))*self.weights).to(in_dtype)


class positionwise_feedforward(nn.Module):

    def __init__(self,d_model,d_ff):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1_weight = nn.init.trunc_normal_(tensor=torch.empty(self.d_ff,self.d_model))
        self.w2_weight = nn.init.trunc_normal_(tensor=torch.empty(self.d_model,self.d_ff))
        self.w3_weight = nn.init.trunc_normal_(tensor=torch.empty(self.d_ff,self.d_model))

    def silu(self,x):
        return torch.sigmoid(x) * x

    def element_wise(self,x,y):
        return torch.einsum("...,...->...",x,y)

    def forward(self,x):
        w3x = torch.einsum("...d,fd->...f",x ,self.w3_weight)
        w1x = torch.einsum("...d,fd->...f",x ,self.w1_weight)

        silu_w1x = self.silu(w1x)

        swiglu_ouptut = self.element_wise(silu_w1x,w3x)

        output = torch.einsum("...f,df->...d", swiglu_ouptut, self.w2_weight)

        return output


class RoPE(nn.Module):

    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = self.theta ** (torch.arange(0,self.d_k,2).float()/d_k)

        positions = torch.arange(max_seq_len).float()

        angles = torch.outer(positions,1.0/freqs)

        self.register_buffer("cos_cached", torch.cos(angles),persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:

        batch_size,seq_len,d_k = x.shape

        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        x_reshaped = x.view(batch_size,seq_len,d_k//2,2)

        x1,x2 = x_reshaped[...,0],x_reshaped[...,1]

        x1_rotated = x1 * cos_pos - x2 * sin_pos
        x2_rotated = x2 * cos_pos + x1 * sin_pos

        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        x_rotated = x_rotated.view(batch_size, seq_len, d_k)
        
        return x_rotated


class Softmax(nn.Module):
    
    def __init__(self, x:torch.Tensor, dimension:int):
        super().__init__()

        self.x = x
        self.dimension = dimension


    def forward(self):
        x_shifted = self.x - torch.max(self.x,dim = self.dimension,keepdim=True)[0]

        exp_x = torch.exp(x_shifted)

        sum_exp_x = torch.sum(exp_x, dim=self.dimension, keepdim=True)

        return exp_x / sum_exp_x

class scaled_dot_product_attention(nn.Module):

    def __init__(self,K,Q,V,mask):
        super().__init__()
        self.Q = Q # [batch,...seq_q,d_k]
        self.K = K # [batch,...,seq_k,d_k]
        self.V = V # [batch,...,seq_k,d_v]
        self.mask = mask # [seq_q,seq_k]
        self.d_k = Q.shape[-1]

    def forward(self):
        scores = torch.einsum("b...qd,b...kd->b...qk",self.Q,self.K)/torch.sqrt(torch.tensor(self.d_k,dtype=torch.float32))
        
        if self.mask is not None:
            scores = torch.where(self.mask,scores,float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.einsum("b...qk,b...kv->b...qv",attention_weights,self.V)

        return output

class multihead_self_attention(nn.Module):

    def __init__(self, d_model, num_heads, use_rope=True):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_k)
        self.output_proj = Linear(num_heads * self.d_v, d_model)

        if self.use_rope:
            self.rope = RoPE(theta=10000, d_k=self.d_k, max_seq_len=1024)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        Q = self.q_proj(x)  # [batch, seq, num_heads * d_k]
        K = self.k_proj(x)  # [batch, seq, num_heads * d_k]
        V = self.v_proj(x)  # [batch, seq, num_heads * d_k]

        # Rearrange to separate heads
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        if self.use_rope:
            # 创建 token_positions [seq_len]
            token_positions = torch.arange(seq_len, device=x.device)

            # 为每个头应用 RoPE，形状 [batch, num_heads, seq, d_k]
            for head in range(self.num_heads):
                Q[:, head, :, :] = self.rope(Q[:, head, :, :], token_positions.unsqueeze(0))
                K[:, head, :, :] = self.rope(K[:, head, :, :], token_positions.unsqueeze(0))

        # 添加因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        # scaled_dot_product_attention 期望 mask True = 允许，False = 屏蔽
        # 但我们的 causal_mask True = 屏蔽，所以需要取反
        allow_mask = ~causal_mask
        allow_mask = allow_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)

        # 使用 scaled_dot_product_attention 类进行计算
        # 注意：参考实现使用 (K, Q, V, mask) 的顺序
        attn = scaled_dot_product_attention(K, Q, V, allow_mask)
        attended_values = attn()  # [batch, num_heads, seq, d_k]

        # Rearrange back to [batch, seq, num_heads * d_k]
        attended_values = rearrange(attended_values, "b h s d -> b s (h d)", h=self.num_heads)

        output = self.output_proj(attended_values)  # [batch, seq, d_model]

        return output
import typing
import torch
import torch.nn as nn
import math
from einops import rearrange
from collections.abc import Iterable
import numpy as np
import os
import typing


class Linear(nn.Module):

    def __init__(self,in_features,out_features,device=None,dtype=None):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std = 2/(self.in_features+self.out_features)

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        return torch.einsum("...i,ji->...j",x,self.weight)


class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):


        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)


    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class rmsnorm(nn.Module):

    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.ones(self.d_model))
        nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

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

        self.w1_weight = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.w2_weight = nn.Parameter(torch.empty(self.d_model, self.d_ff))
        self.w3_weight = nn.Parameter(torch.empty(self.d_ff, self.d_model))

        nn.init.trunc_normal_(self.w1_weight, mean=0, std=1, a=-3, b=3)
        nn.init.trunc_normal_(self.w2_weight, mean=0, std=1, a=-3, b=3)
        nn.init.trunc_normal_(self.w3_weight, mean=0, std=1, a=-3, b=3)

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

    def __init__(self, d_model, num_heads, use_rope=True,max_seq_len=1024, theta=10000,token_positions=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.token_positions = token_positions

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_k)
        self.output_proj = Linear(num_heads * self.d_v, d_model)

        if self.use_rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
    
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
            # token_positions = self.token_positions

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


class transformer_block(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,use_rope=True,max_seq_len=1024,theta=10000):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm1 = rmsnorm(d_model = self.d_model)
        self.norm2 = rmsnorm(d_model = self.d_model)
        self.attn = multihead_self_attention(d_model = self.d_model, num_heads = self.num_heads, max_seq_len=max_seq_len, theta=theta,use_rope=use_rope)
        self.ffn = positionwise_feedforward(d_model = self.d_model, d_ff = self.d_ff)


    def forward(self,x):

        block1_output = x + self.attn(self.norm1(x))
        block2_output = block1_output + self.ffn(self.norm2(block1_output))

        return block2_output


class transformer_lm(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,vocab_size,context_length,num_layers,use_rope,max_seq_len=1024,theta=10000):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.Token_Embedding = Embedding(num_embeddings=self.vocab_size, embedding_dim = self.d_model)
        self.layers = nn.ModuleList([transformer_block(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, use_rope=use_rope,max_seq_len=self.max_seq_len, theta=self.theta) for _ in range(self.num_layers)])
        self.norm = rmsnorm(d_model = self.d_model)
        self.linear = Linear(in_features=self.d_model, out_features=self.vocab_size)


    def forward(self,x):

        x = self.Token_Embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        x = self.linear(x)

        # softmax_layer = Softmax(x, dimension=-1)
        # return softmax_layer.forward()

        return x


def cross_entropy(inputs,targets):
    
    vocab_size = inputs.shape[-1]
    max_logits = torch.max(inputs,dim=-1,keepdim=True)[0]

    inputs_shifted = inputs - max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_shifted),dim=-1))
    correct_logits = torch.gather(inputs_shifted, -1, targets.unsqueeze(-1)).squeeze(-1) 

    losses = -correct_logits + log_sum_exp

    return torch.mean(losses)


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state.setdefault('t', 0)
                state.setdefault('m_t', torch.zeros_like(p.data))
                state.setdefault('v_t', torch.zeros_like(p.data))

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            epsilon = group['eps']
            lambda_ = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                t = state.get("t", 0) + 1

                m_t = state.get("m_t", torch.zeros_like(p.data))
                v_t = state.get("v_t", torch.zeros_like(p.data))

                grad = p.grad.data

                m_t = beta_1 * m_t + (1 - beta_1) * grad
                v_t = beta_2 * v_t + (1 - beta_2) * (grad * grad)

                state["t"] = t
                state["m_t"] = m_t
                state["v_t"] = v_t

                bias_correction1 = 1 - (beta_1 ** t)
                bias_correction2 = 1 - (beta_2 ** t)

                alpha_t = lr * (math.sqrt(bias_correction2) / bias_correction1)

                denom = v_t.sqrt().add(epsilon)
                update = m_t / denom
                p.data = p.data - alpha_t * update

                if lambda_ != 0:
                    p.data = p.data - lr * lambda_ * p.data

        return loss

def silu(x):
    return torch.sigmoid(x) * x

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):

    if t < T_w:
        alpha_t = (t / T_w) * alpha_max

    elif T_w <= t <= T_c:
        cos_num = ((t - T_w)/(T_c - T_w)) * torch.pi
        alpha_t = alpha_min + 1/2 * (1 + math.cos(cos_num)) * (alpha_max - alpha_min)

    else:
        alpha_t = alpha_min

    return alpha_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float,eps = 1e-6):
    grads = []

    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.data.view(-1))

    if not grads:
        return torch.tensor(0.0)

    all_grads = torch.cat(grads)
    l2_norm = torch.norm(all_grads,p=2)


    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + eps)

        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

def data_loading(x:np.array,batch_size,context_length,device='cpu'):
    max_start_idx = len(x) - context_length

    start_indices = np.random.randint(0,max_start_idx,size=batch_size)

    inputs = np.zeros((batch_size,context_length),dtype=np.int64)
    targets = np.zeros((batch_size,context_length),dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        inputs[i] = x[start_idx:start_idx + context_length]
        targets[i] = x[start_idx + 1:start_idx + context_length + 1]

    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)

    return inputs_tensor,targets_tensor


def save_checkpoint(model:torch.nn.Module,optimizer:torch.optim.Optimizer,iteration:int,out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'iteration': iteration
    }

    torch.save(checkpoint,out)


def load_checkpoint(src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model:torch.nn.Module, optimizer:torch.optim.Optimizer):

    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']


def decoding(model: torch.nn.Module,
            prompt_tokens:list[int],
            max_tokens:int,
            temperature:float,
            top_p:float,
            end_token_id:int):

    model.eval()
    device = next(model.parameters()).device

    generated = prompt_tokens.copy()

    with torch.no_grad():
        for _ in range(max_tokens):

            input_tensor = torch.tensor([generated],dtype=torch.long).to(device)

            logits = model(input_tensor)

            next_token_logits = logits[:,-1,:]

            next_token_logits = next_token_logits / temperature

            probs = torch.softmax(next_token_logits, dim=-1)

            if top_p < 1.0:
                probs = apply_top_p_sampling(probs, top_p)

            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()

            generated.append(next_token_id)


            if next_token_id == end_token_id:
                break

    return generated

def apply_top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:

    batch_size, vocab_size = probs.shape
    
    for i in range(batch_size):
        # 对第i个batch的概率排序
        sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到需要保留的token数量 (至少保留1个)
        cutoff_mask = cumulative_probs <= top_p
        cutoff_mask[0] = True  # 确保至少保留概率最高的token
        
        # 截断概率并重新归一化
        top_p_probs = sorted_probs * cutoff_mask
        top_p_probs = top_p_probs / top_p_probs.sum()
        
        # 将截断后的概率放回原位置
        truncated_probs = torch.zeros_like(probs[i])
        truncated_probs[sorted_indices] = top_p_probs
        probs[i] = truncated_probs
    
    return probs

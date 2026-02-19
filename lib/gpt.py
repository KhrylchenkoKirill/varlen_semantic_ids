import os
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch import Tensor, nn


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()
        
    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device='cuda')
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device='cuda')
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = 128 * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


class CausalSelfAttention(nn.Module):
    def __init__(
            self, 
            dim: int, 
            head_dim: int, 
            num_heads: int, 
            causal=True
    ):
        super().__init__()
        self.causal = causal

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim

        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        self.qkvo_w.label = 'attn'
        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()

    
    def forward(self, x: Tensor, cos_sin, scale):
        B, T, D = x.shape

        q, k, v = self._qkv(x, cos_sin)  # [B, T, nH, Hd]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, nH, T, Hd]

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, scale=scale
        )  # [B, nH, T, Hd]
        y = y.transpose(1, 2)  # [B, T, nH, Hd]
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)  # [B, T, D]
        y = F.linear(
            y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y)
        )  # [B, T, D]
        return y
    
    def _qkv(self, x: Tensor, cos_sin):
        B, T, D = x.shape
        q, k, v = F.linear(
            x, self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)

        cos, sin = cos_sin
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)

        return q, k, v  # [B, T, nH, Hd]


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make matrices the same shape to enable batched call in optimizer
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        # label modules to enable custom optimizer sizing
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.type_as(x))
        return x


class Block(nn.Module):
    def __init__(
            self, 
            dim: int, 
            head_dim: int, 
            num_heads: int,
            causal=True,
            dropout=0.
    ):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, causal=causal)
        self.mlp  = MLP(dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, cos_sin, scale):
        x = x + self.dropout(self.attn(norm(x), cos_sin, scale))
        x = x + self.dropout(self.mlp(norm(x)))
        return x


class GPT(nn.Module):
    def __init__(
            self, 
            num_layers: int, 
            num_heads: int, 
            model_dim: int, 
            max_seq_len: int,
            embeddings_cls=None,
            causal=True,
            dropout=0.
    ):
        super().__init__()
        head_dim = model_dim // num_heads

        self.embed = None
        if embeddings_cls is not None: 
            self.embed = embeddings_cls()
        self.yarn = Yarn(head_dim, max_seq_len)
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads, causal=causal, dropout=dropout)
            for i in range(num_layers)
        ])

        self.L = num_layers
    
    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        x = norm(x)
        for i in range(self.L):
            x = self.blocks[i](x, (self.yarn.cos, self.yarn.sin), self.yarn.attn_scale)
        x = norm(x)
        return x

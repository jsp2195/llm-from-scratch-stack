from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llmstack.model.rotary import apply_rope


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_enabled: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope_enabled = rope_enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope_enabled:
            q, k = apply_rope(q), apply_rope(k)
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
            mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
            att = torch.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm = RMSNorm if cfg.norm_type == 'rmsnorm' else nn.LayerNorm
        self.norm1 = norm(cfg.d_model)
        self.norm2 = norm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_enabled)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

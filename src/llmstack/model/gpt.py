from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from llmstack.model.layers import RMSNorm, TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model) if cfg.norm_type == 'rmsnorm' else nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        scale = 1 / math.sqrt(2 * cfg.n_layers)
        for b in self.blocks:
            b.attn.out.weight.data.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'weight'):
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None):
        b, t = input_ids.shape
        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        logits = self.lm_head(self.norm(x))
        if labels is None:
            return logits, None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().clone()
        if loss_mask is not None:
            shift_mask = loss_mask[:, 1:].to(torch.bool)
            shift_labels = shift_labels.masked_fill(~shift_mask, -100)
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100
        )
        return logits, loss

    def sequence_logprob(self, input_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self(input_ids)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (tok_logp * target_mask[:, 1:]).sum(dim=-1)

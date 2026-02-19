"""Decoder-only GPT model."""

from __future__ import annotations

import torch
from torch import nn

from llmstack.model.layers import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.tok_emb = nn.Embedding(m.vocab_size, m.d_model)
        self.pos_emb = nn.Embedding(m.max_seq_len, m.d_model)
        self.drop = nn.Dropout(m.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(m.d_model, m.n_heads, m.d_ff, m.dropout, m.norm_type) for _ in range(m.n_layers)]
        )
        self.ln_f = nn.LayerNorm(m.d_model)
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        b, t = input_ids.shape
        pos = torch.arange(0, t, device=input_ids.device, dtype=torch.long)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0
            )
        return {"logits": logits, "loss": loss}

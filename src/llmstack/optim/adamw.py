from __future__ import annotations

import torch


def build_adamw(model, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith('bias') or 'norm' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {'params': decay, 'weight_decay': cfg.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

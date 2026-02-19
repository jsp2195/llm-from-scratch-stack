"""Optimizer wrapper."""

import torch


def build_adamw(params, cfg):
    return torch.optim.AdamW(
        params,
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

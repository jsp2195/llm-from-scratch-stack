"""Schedulers."""

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, cfg):
    total = cfg.train.max_steps
    warmup = max(1, cfg.sched.warmup_steps)

    def fn(step: int):
        if cfg.sched.name == "linear":
            return max(0.0, 1 - step / max(total, 1))
        if cfg.sched.name == "cosine":
            return cfg.sched.min_lr / cfg.optim.lr + 0.5 * (1 + math.cos(math.pi * step / total))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        floor = cfg.sched.min_lr / cfg.optim.lr
        return floor + (1 - floor) * cos

    return LambdaLR(optimizer, lr_lambda=fn)

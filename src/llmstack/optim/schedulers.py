from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, cfg, max_steps: int):
    def cosine(step):
        return cfg.min_lr + (1 - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * step / max_steps))

    def linear(step):
        return max(cfg.min_lr, 1 - step / max_steps)

    def warmup_cos(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        p = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.min_lr + (1 - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    fn = {'cosine': cosine, 'linear': linear, 'warmup_cosine': warmup_cos}[cfg.name]
    return LambdaLR(optimizer, fn)

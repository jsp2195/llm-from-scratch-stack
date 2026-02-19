from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group['betas']
            lr, wd = group['lr'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st['m'] = torch.zeros_like(p)
                m = st['m']
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = (m * b1 + g * (1 - b1)).sign()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss

from __future__ import annotations

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_rank0() -> bool:
    return rank() == 0


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    x = all_reduce_sum(x)
    x /= world_size()
    return x

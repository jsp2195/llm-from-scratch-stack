"""Distributed helpers."""

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_rank0() -> bool:
    return rank() == 0


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(x)
        x /= world_size()
    return x

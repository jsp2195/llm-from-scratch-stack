"""DDP setup utilities."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(backend: str = "nccl") -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def wrap_ddp(model, find_unused: bool = False):
    device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids, find_unused_parameters=find_unused)

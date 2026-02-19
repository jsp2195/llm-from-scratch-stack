from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(backend: str = 'nccl'):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def wrap_ddp(model, find_unused_params: bool = False):
    if dist.is_initialized():
        return DDP(model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None, find_unused_parameters=find_unused_params)
    return model

from __future__ import annotations

from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def wrap_fsdp(model, mixed_precision: bool = False, cpu_offload: bool = False):
    mp = MixedPrecision(param_dtype=None, reduce_dtype=None, buffer_dtype=None) if mixed_precision else None
    off = CPUOffload(offload_params=True) if cpu_offload else None
    return FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1_000_000), mixed_precision=mp, cpu_offload=off)

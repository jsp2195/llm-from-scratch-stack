"""Optional FSDP wrapping."""

import torch


def maybe_wrap_fsdp(model, cfg):
    if cfg.dist.mode != "fsdp":
        return model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model)
    except Exception as exc:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"FSDP unavailable, falling back to unwrapped model: {exc}")
        return model

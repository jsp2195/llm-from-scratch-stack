"""AMP helpers."""

from contextlib import nullcontext

import torch


def autocast_context(device: str, precision: str):
    if device == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def build_scaler(precision: str):
    return torch.cuda.amp.GradScaler(enabled=precision == "fp16")

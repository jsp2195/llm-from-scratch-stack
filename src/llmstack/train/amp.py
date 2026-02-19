from contextlib import nullcontext

import torch


def autocast_context(precision: str, device: str):
    if precision == 'fp16' and device == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    if precision == 'bf16' and device == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def make_scaler(precision: str, device: str):
    return torch.cuda.amp.GradScaler() if precision == 'fp16' and device == 'cuda' else None

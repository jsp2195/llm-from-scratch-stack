"""Environment detection."""

import torch


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

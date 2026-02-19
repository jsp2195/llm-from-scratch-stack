import torch


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

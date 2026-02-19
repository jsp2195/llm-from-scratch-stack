"""Optional rotary embeddings."""

import torch


def apply_rotary(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return q, k

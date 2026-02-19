from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    b, h, t, d = x.shape
    half = d // 2
    freqs = torch.arange(half, device=x.device).float() / half
    pos = torch.arange(t, device=x.device).float()
    angles = pos[:, None] * (10000 ** (-freqs))[None, :]
    cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half : half * 2]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., half * 2 :]], dim=-1)
    return out

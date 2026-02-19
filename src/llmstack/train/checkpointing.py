from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: str, state: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    os.close(fd)
    torch.save(state, tmp)
    os.replace(tmp, path)


def capture_rng() -> dict:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(rng: dict):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if torch.cuda.is_available() and rng.get('cuda') is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])

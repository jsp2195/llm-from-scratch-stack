"""Preference dataset for DPO."""

import json
from pathlib import Path

from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, path: str):
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]

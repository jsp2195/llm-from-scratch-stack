"""SFT training utilities."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int, sep: str):
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sep = sep

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        prompt = row["prompt"]
        response = row["response"]
        full = prompt + self.sep + response
        ids = self.tokenizer.encode(full)[: self.seq_len + 1]
        if len(ids) < self.seq_len + 1:
            ids += [self.tokenizer.pad_id] * (self.seq_len + 1 - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        prompt_len = len(self.tokenizer.encode(prompt + self.sep)) - 1
        y[: max(0, prompt_len - 1)] = -100
        return {"input_ids": x, "labels": y}

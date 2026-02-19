from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset, IterableDataset


class TextMapDataset(Dataset):
    def __init__(self, path: str, fmt: str, text_field: str = "text"):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        self.samples = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            self.samples.append(_extract_text(line, fmt, text_field))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


class TextIterableDataset(IterableDataset):
    def __init__(self, path: str, fmt: str, text_field: str = "text", shuffle: bool = False, seed: int = 0):
        self.path = path
        self.fmt = fmt
        self.text_field = text_field
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        buf = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                text = _extract_text(line, self.fmt, self.text_field)
                if self.shuffle:
                    buf.append(text)
                    if len(buf) >= 64:
                        rng.shuffle(buf)
                        while buf:
                            yield buf.pop()
                else:
                    yield text
        if buf:
            rng.shuffle(buf)
            while buf:
                yield buf.pop()


def _extract_text(line: str, fmt: str, text_field: str) -> str:
    if fmt == "txt":
        return line.rstrip("\n")
    if fmt == "jsonl":
        return json.loads(line)[text_field]
    raise ValueError(f"unsupported format {fmt}")


def collate_token_blocks(batch: list[str], tokenizer, seq_len: int, add_bos: bool, add_eos: bool, pad_to_seq_len: bool):
    ids_list = []
    masks = []
    for text in batch:
        ids = tokenizer.encode(text)
        if add_bos:
            ids = [tokenizer.bos_id] + ids
        if add_eos:
            ids = ids + [tokenizer.eos_id]
        ids = ids[:seq_len]
        length = len(ids)
        if pad_to_seq_len and length < seq_len:
            ids = ids + [tokenizer.pad_id] * (seq_len - length)
        mask = [1] * length + [0] * (seq_len - length)
        ids_list.append(ids)
        masks.append(mask)
    x = torch.tensor(ids_list, dtype=torch.long)
    m = torch.tensor(masks, dtype=torch.float32)
    labels = x.clone()
    labels[m == 0] = -100
    return {"input_ids": x, "labels": labels, "loss_mask": m}

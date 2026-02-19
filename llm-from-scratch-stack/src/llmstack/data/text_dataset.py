"""Text datasets supporting streaming and map-style loading."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset

from llmstack.data.packing import pack_tokens_to_blocks


class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int, fmt: str = "jsonl", text_field: str = "text"):
        self.samples = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if fmt == "jsonl":
                self.samples.append(json.loads(line)[text_field])
            else:
                self.samples.append(line)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer.encode(self.samples[idx])
        tokens = tokens[: self.seq_len + 1]
        if len(tokens) < self.seq_len + 1:
            tokens += [self.tokenizer.pad_id] * (self.seq_len + 1 - len(tokens))
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class StreamingTextDataset(IterableDataset):
    def __init__(
        self, path: str, tokenizer, seq_len: int, fmt: str = "jsonl", text_field: str = "text"
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.fmt = fmt
        self.text_field = text_field

    def _iter_tokens(self) -> Iterator[int]:
        for line in Path(self.path).open(encoding="utf-8"):
            if not line.strip():
                continue
            text = json.loads(line)[self.text_field] if self.fmt == "jsonl" else line.strip()
            yield from self.tokenizer.encode(text)
            yield self.tokenizer.eos_id

    def __iter__(self):
        for block in pack_tokens_to_blocks(self._iter_tokens(), self.seq_len + 1):
            yield {"input_ids": block[:-1], "labels": block[1:]}

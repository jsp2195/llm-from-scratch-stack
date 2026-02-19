from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler

from llmstack.data.text_dataset import TextIterableDataset, TextMapDataset, collate_token_blocks


class DataModule:
    def __init__(self, cfg, tokenizer, world_size: int = 1, rank: int = 0):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.world_size = world_size
        self.rank = rank
        self.train_sampler = None

    def train_dataloader(self):
        c = self.cfg.data
        if c.streaming:
            ds = TextIterableDataset(c.train_path, c.format, c.text_field, c.shuffle, self.cfg.train.seed + self.rank)
            sampler = None
            shuffle = False
        else:
            ds = TextMapDataset(c.train_path, c.format, c.text_field)
            sampler = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, shuffle=c.shuffle) if self.world_size > 1 else None
            self.train_sampler = sampler
            shuffle = c.shuffle and sampler is None
        return DataLoader(
            ds,
            batch_size=self.cfg.train.micro_batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=c.num_workers,
            pin_memory=self.cfg.train.device == "cuda",
            persistent_workers=c.num_workers > 0,
            collate_fn=partial(
                collate_token_blocks,
                tokenizer=self.tokenizer,
                seq_len=c.seq_len,
                add_bos=c.add_bos,
                add_eos=c.add_eos,
                pad_to_seq_len=c.pad_to_seq_len,
            ),
        )

    def val_dataloader(self):
        c = self.cfg.data
        ds = TextMapDataset(c.val_path, c.format, c.text_field)
        sampler = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, shuffle=False) if self.world_size > 1 else None
        return DataLoader(
            ds,
            batch_size=self.cfg.train.micro_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=c.num_workers,
            pin_memory=self.cfg.train.device == "cuda",
            persistent_workers=c.num_workers > 0,
            collate_fn=partial(
                collate_token_blocks,
                tokenizer=self.tokenizer,
                seq_len=c.seq_len,
                add_bos=c.add_bos,
                add_eos=c.add_eos,
                pad_to_seq_len=c.pad_to_seq_len,
            ),
        )

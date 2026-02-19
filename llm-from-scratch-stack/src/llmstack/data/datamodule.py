"""Dataloader builder."""

from torch.utils.data import DataLoader

from llmstack.data.text_dataset import StreamingTextDataset, TextDataset


def build_dataloaders(cfg, tokenizer, distributed: bool = False):
    if cfg.data.streaming:
        train_ds = StreamingTextDataset(
            cfg.data.train_path, tokenizer, cfg.data.seq_len, cfg.data.format, cfg.data.text_field
        )
        val_ds = StreamingTextDataset(
            cfg.data.val_path, tokenizer, cfg.data.seq_len, cfg.data.format, cfg.data.text_field
        )
        sampler = None
    else:
        train_ds = TextDataset(
            cfg.data.train_path, tokenizer, cfg.data.seq_len, cfg.data.format, cfg.data.text_field
        )
        val_ds = TextDataset(cfg.data.val_path, tokenizer, cfg.data.seq_len, cfg.data.format, cfg.data.text_field)
        sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.micro_batch_size,
        shuffle=(cfg.data.shuffle and sampler is None and not cfg.data.streaming),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.micro_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return train_loader, val_loader

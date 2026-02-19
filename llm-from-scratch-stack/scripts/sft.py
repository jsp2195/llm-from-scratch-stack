#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from llmstack.model.gpt import GPTModel
from llmstack.optim.adamw import build_adamw
from llmstack.optim.schedulers import build_scheduler
from llmstack.posttrain.sft_trainer import SFTDataset
from llmstack.tokenization.tokenizer import TokenizerWrapper
from llmstack.train.checkpointing import load_checkpoint
from llmstack.train.engine import Trainer
from llmstack.utils.env import get_default_device
from llmstack.utils.logging import RunLogger


@hydra.main(version_base=None, config_path="../configs", config_name="sft")
def main(cfg):
    run_dir = Path(cfg.train.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_sft"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.run_dir = str(run_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    tok = TokenizerWrapper.from_dir(cfg.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size
    device = get_default_device() if cfg.train.device == "cuda" else cfg.train.device
    model = GPTModel(cfg).to(device)
    ckpt = load_checkpoint(cfg.posttrain.sft.base_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    ds = SFTDataset(cfg.posttrain.sft.data_path, tok, cfg.data.seq_len, cfg.posttrain.sft.separator)
    dl = DataLoader(ds, batch_size=cfg.train.micro_batch_size, shuffle=True)
    val = DataLoader(ds, batch_size=cfg.train.micro_batch_size)
    optimizer = build_adamw(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger = RunLogger(str(run_dir))
    trainer = Trainer(cfg, model, optimizer, scheduler, dl, val, logger, device)
    trainer.fit()


if __name__ == "__main__":
    main()

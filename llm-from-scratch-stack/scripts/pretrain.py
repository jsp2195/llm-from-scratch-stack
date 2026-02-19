#!/usr/bin/env python
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from llmstack.data.datamodule import build_dataloaders
from llmstack.model.gpt import GPTModel
from llmstack.optim.adamw import build_adamw
from llmstack.optim.lion import Lion
from llmstack.optim.schedulers import build_scheduler
from llmstack.tokenization.tokenizer import TokenizerWrapper
from llmstack.train.engine import Trainer
from llmstack.utils.env import get_default_device
from llmstack.utils.logging import RunLogger
from llmstack.utils.manifest import write_manifest
from llmstack.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg):
    seed_everything(cfg.train.seed)
    run_dir = Path(cfg.train.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_pretrain"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.run_dir = str(run_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    write_manifest(str(run_dir), cfg, sys.argv)

    tokenizer = TokenizerWrapper.from_dir(cfg.tokenizer_dir)
    cfg.model.vocab_size = tokenizer.vocab_size
    device = get_default_device() if cfg.train.device == "cuda" else cfg.train.device
    model = GPTModel(cfg).to(device)
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    if cfg.optim.name == "lion":
        optimizer = Lion(
            model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay
        )
    else:
        optimizer = build_adamw(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger = RunLogger(str(run_dir))
    trainer = Trainer(cfg, model, optimizer, scheduler, train_loader, val_loader, logger, device)
    trainer.fit()
    logger.close()


if __name__ == "__main__":
    main()

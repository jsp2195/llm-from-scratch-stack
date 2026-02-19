#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from llmstack.model.gpt import GPTModel
from llmstack.posttrain.dpo_trainer import DPOTrainer
from llmstack.posttrain.preference_dataset import PreferenceDataset
from llmstack.tokenization.tokenizer import TokenizerWrapper
from llmstack.train.checkpointing import load_checkpoint, save_checkpoint
from llmstack.utils.env import get_default_device


@hydra.main(version_base=None, config_path="../configs", config_name="dpo")
def main(cfg):
    run_dir = Path(cfg.train.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_dpo"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    tok = TokenizerWrapper.from_dir(cfg.tokenizer_dir)
    cfg.model.vocab_size = tok.vocab_size
    device = get_default_device() if cfg.train.device == "cuda" else cfg.train.device
    model = GPTModel(cfg).to(device)
    ckpt = load_checkpoint(cfg.posttrain.dpo.base_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    trainer = DPOTrainer(model, cfg.posttrain.dpo.beta)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
    ds = PreferenceDataset(cfg.posttrain.dpo.data_path)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    for step, row in enumerate(dl):
        if step >= cfg.train.max_steps:
            break
        chosen = torch.tensor([tok.encode(row["prompt"][0] + row["chosen"][0])], device=device)
        rejected = torch.tensor([tok.encode(row["prompt"][0] + row["rejected"][0])], device=device)
        loss = trainer.loss(chosen, rejected)
        optim.zero_grad()
        loss.backward()
        optim.step()
    save_checkpoint(str(run_dir / "dpo_last.pt"), {"model": model.state_dict(), "config": OmegaConf.to_container(cfg)})
    (run_dir / "dpo_metrics.json").write_text(json.dumps({"steps": step + 1}), encoding="utf-8")


if __name__ == "__main__":
    main()

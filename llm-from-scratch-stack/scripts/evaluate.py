#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from llmstack.data.datamodule import build_dataloaders
from llmstack.eval.perplexity import evaluate_perplexity
from llmstack.eval.probes import bracket_matching_probe, repeat_after_me_probe, simple_addition_probe
from llmstack.model.gpt import GPTModel
from llmstack.tokenization.tokenizer import TokenizerWrapper
from llmstack.train.checkpointing import load_checkpoint
from llmstack.utils.env import get_default_device


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    run_dir = Path(cfg.train.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    tokenizer = TokenizerWrapper.from_dir(cfg.tokenizer_dir)
    cfg.model.vocab_size = tokenizer.vocab_size
    device = get_default_device() if cfg.train.device == "cuda" else cfg.train.device
    model = GPTModel(cfg).to(device)
    ckpt = load_checkpoint(cfg.eval.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    _, val_loader = build_dataloaders(cfg, tokenizer)
    report = evaluate_perplexity(model, val_loader, device, cfg.eval.max_batches)
    report.update(repeat_after_me_probe(model, tokenizer, device))
    report.update(bracket_matching_probe(model, tokenizer, device))
    report.update(simple_addition_probe(model, tokenizer, device))
    out = run_dir / cfg.eval.report_name
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

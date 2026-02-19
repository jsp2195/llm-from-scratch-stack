#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from llmstack.data.datamodule import DataModule
from llmstack.eval.perplexity import evaluate_perplexity
from llmstack.eval.probes import run_probes
from llmstack.model.gpt import GPTModel
from llmstack.tokenization.tokenizer import Tokenizer


@hydra.main(version_base=None, config_path='../configs', config_name='eval')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size
    model = GPTModel(cfg.model).to(cfg.train.device)
    ckpt = torch.load(cfg.eval.checkpoint_path, map_location=cfg.train.device)
    model.load_state_dict(ckpt['model'])
    dm = DataModule(cfg, tok)
    ppl = evaluate_perplexity(model, dm.val_dataloader(), cfg.train.device, cfg.eval.max_batches)
    probes = run_probes(model, tok, cfg.train.device)
    report = {'perplexity': ppl, 'probes': probes}
    out = Path(cfg.eval.checkpoint_path).parent / 'eval_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

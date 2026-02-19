#!/usr/bin/env python
from __future__ import annotations

import json
import os

import hydra
import torch
from omegaconf import DictConfig

from llmstack.model.gpt import GPTModel
from llmstack.posttrain.preference_dataset import load_jsonl
from llmstack.posttrain.sft_trainer import sft_step
from llmstack.tokenization.tokenizer import Tokenizer


@hydra.main(version_base=None, config_path='../configs', config_name='sft')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size
    model = GPTModel(cfg.model).to(cfg.train.device)
    ckpt = torch.load(cfg.sft.base_checkpoint, map_location=cfg.train.device)
    model.load_state_dict(ckpt['model'])
    data = load_jsonl(cfg.sft.data_path)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr))
    model.train()
    for i in range(cfg.sft.max_steps):
        loss = sft_step(model, tok, [data[i % len(data)]], cfg.sft.separator, cfg.train.device)
        opt.zero_grad(); loss.backward(); opt.step()
    out = cfg.sft.base_checkpoint.replace('.ckpt', '.sft.ckpt')
    torch.save({'model': model.state_dict()}, out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
from __future__ import annotations

import copy
import os

import hydra
import torch
from omegaconf import DictConfig

from llmstack.eval.probes import run_probes
from llmstack.model.gpt import GPTModel
from llmstack.posttrain.dpo_trainer import dpo_loss
from llmstack.posttrain.preference_dataset import load_jsonl
from llmstack.tokenization.tokenizer import Tokenizer


@hydra.main(version_base=None, config_path='../configs', config_name='dpo')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size
    policy = GPTModel(cfg.model).to(cfg.train.device)
    ckpt = torch.load(cfg.dpo.base_checkpoint, map_location=cfg.train.device)
    policy.load_state_dict(ckpt['model'])
    ref = GPTModel(cfg.model).to(cfg.train.device)
    ref.load_state_dict(ckpt['model']) if cfg.dpo.reference_mode == 'frozen_base' else ref.load_state_dict(copy.deepcopy(policy.state_dict()))
    ref.eval(); [p.requires_grad_(False) for p in ref.parameters()]
    data = load_jsonl(cfg.dpo.data_path)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(cfg.optim.lr))
    for i in range(cfg.dpo.max_steps):
        loss = dpo_loss(policy, ref, tok, data[i % len(data)], float(cfg.dpo.beta), cfg.train.device)
        opt.zero_grad(); loss.backward(); opt.step()
    out = cfg.dpo.base_checkpoint.replace('.ckpt', '.dpo.ckpt')
    torch.save({'model': policy.state_dict(), 'probes': run_probes(policy, tok, cfg.train.device)}, out)


if __name__ == '__main__':
    main()

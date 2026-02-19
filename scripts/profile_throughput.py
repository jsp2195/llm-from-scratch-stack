#!/usr/bin/env python
from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig

from llmstack.model.gpt import GPTModel


@hydra.main(version_base=None, config_path='../configs', config_name='pretrain')
def main(cfg: DictConfig):
    m = GPTModel(cfg.model).to(cfg.train.device)
    x = torch.randint(0, cfg.model.vocab_size, (cfg.train.micro_batch_size, cfg.data.seq_len), device=cfg.train.device)
    t0 = time.time()
    for _ in range(20):
        logits, _ = m(x, labels=x)
        logits.sum().backward()
    dt = time.time() - t0
    toks = 20 * cfg.train.micro_batch_size * cfg.data.seq_len
    mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    print({'tokens_per_sec': toks / dt, 'step_time': dt / 20, 'peak_memory': mem})


if __name__ == '__main__':
    main()

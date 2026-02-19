#!/usr/bin/env python
from __future__ import annotations

import time

import hydra
import torch

from llmstack.model.gpt import GPTModel


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(cfg).to(device)
    x = torch.randint(0, cfg.model.vocab_size, (cfg.train.micro_batch_size, cfg.data.seq_len), device=device)
    y = x.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = 20
    t0 = time.time()
    for _ in range(steps):
        loss = model(x, y)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    dt = time.time() - t0
    toks = steps * x.numel()
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print({"tokens_per_sec": toks / dt, "step_time": dt / steps, "peak_gb": peak})


if __name__ == "__main__":
    main()

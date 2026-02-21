#!/usr/bin/env python
from __future__ import annotations

import os
import random
import hydra
import torch

from omegaconf import DictConfig

from llmstack.model.gpt import GPTModel
from llmstack.posttrain.preference_dataset import load_jsonl
from llmstack.posttrain.sft_trainer import sft_step
from llmstack.tokenization.tokenizer import Tokenizer


# -------------------------------------------------
# Batch sampler
# -------------------------------------------------

def sample_batch(data, batch_size, rng):
    idxs = [rng.randrange(len(data)) for _ in range(batch_size)]
    return [data[i] for i in idxs]


# -------------------------------------------------
# Trainer
# -------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="sft")
def main(cfg: DictConfig):

    os.chdir(hydra.utils.get_original_cwd())

    device = cfg.train.device
    rng = random.Random(cfg.train.seed)

    # ---------- tokenizer ----------
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size

    # ---------- model ----------
    model = GPTModel(cfg.model).to(device)

    ckpt = torch.load(cfg.sft.base_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    model.train()

    # ---------- data ----------
    data = load_jsonl(cfg.sft.data_path)

    # ---------- optimizer ----------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.get("weight_decay", 0.01)),
    )

    # ---------- training params ----------
    batch_size = int(cfg.sft.get("batch_size", 8))
    max_steps = int(cfg.sft.max_steps)
    separator = cfg.sft.separator

    # ---------- loop ----------
    for step in range(max_steps):

        batch = sample_batch(data, batch_size, rng)

        loss = sft_step(
            model,
            tok,
            batch,
            separator,
            device,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping stabilizes small GPTs
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        if step % 50 == 0:
            print(f"SFT step {step:05d} | loss {loss.item():.4f}")

    # ---------- save ----------
    base = os.path.splitext(cfg.sft.base_checkpoint)[0]
    out = base + ".sft.ckpt"

    torch.save({"model": model.state_dict()}, out)

    print(f"\nSaved SFT checkpoint: {out}")


if __name__ == "__main__":
    main()

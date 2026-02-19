from __future__ import annotations

import torch


def sft_step(model, tokenizer, batch, separator: str, device: str):
    losses = []
    for ex in batch:
        prompt, resp = ex['prompt'], ex['response']
        full = prompt + separator + resp
        ids = tokenizer.encode(full)
        cut = len(tokenizer.encode(prompt + separator))
        x = torch.tensor([ids], dtype=torch.long, device=device)
        labels = x.clone()
        labels[:, :cut] = -100
        _, loss = model(x, labels=labels)
        losses.append(loss)
    return torch.stack(losses).mean()

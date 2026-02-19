"""Perplexity evaluation."""

import math

import torch

from llmstack.utils.distributed import all_reduce_mean


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device: str, max_batches: int = 50) -> dict:
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        out = model(batch["input_ids"].to(device), batch["labels"].to(device))
        losses.append(out["loss"].detach())
    loss = torch.stack(losses).mean() if losses else torch.tensor(float("nan"), device=device)
    loss = all_reduce_mean(loss)
    return {"val_loss": float(loss.item()), "perplexity": float(math.exp(loss.item()))}

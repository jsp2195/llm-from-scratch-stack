from __future__ import annotations

import math

import torch


def evaluate_perplexity(model, loader, device: str, max_batches: int = 50):
    total_nll = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['loss_mask'].to(device)
        _, loss = model(ids, labels=labels, loss_mask=mask)
        toks = int(mask[:, 1:].sum().item())
        total_nll += float(loss.item()) * toks
        total_tokens += toks
    loss = total_nll / max(1, total_tokens)
    return {'loss': loss, 'ppl': math.exp(min(20, loss)), 'tokens': total_tokens}

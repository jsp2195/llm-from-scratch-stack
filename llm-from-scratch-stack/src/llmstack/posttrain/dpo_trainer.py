from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(policy, ref, tokenizer, ex, beta: float, device: str):
    def seq_lp(model, text):
        ids = tokenizer.encode(text)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(x, dtype=torch.float32)
        return model.sequence_logprob(x, mask)

    p_c = seq_lp(policy, ex['prompt'] + ex['chosen'])
    p_r = seq_lp(policy, ex['prompt'] + ex['rejected'])
    with torch.no_grad():
        r_c = seq_lp(ref, ex['prompt'] + ex['chosen'])
        r_r = seq_lp(ref, ex['prompt'] + ex['rejected'])
    delta = (p_c - p_r) - (r_c - r_r)
    return -F.logsigmoid(beta * delta).mean()

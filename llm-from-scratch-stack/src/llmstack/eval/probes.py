"""Offline capability probes."""

from __future__ import annotations

import torch


def repeat_after_me_probe(model, tokenizer, device: str) -> dict:
    prompt = "repeat after me: alpha beta gamma"
    ids = tokenizer.encode(prompt)
    out = model(torch.tensor([ids], device=device))
    pred = out["logits"][0, -1].argmax().item()
    return {"repeat_after_me_next_token": int(pred)}


def bracket_matching_probe(model, tokenizer, device: str) -> dict:
    ids = tokenizer.encode("([{}")
    out = model(torch.tensor([ids], device=device))
    logits = out["logits"][0, -1]
    close_ids = tokenizer.encode(")]", add_special_tokens=False)
    close = close_ids[0] if close_ids else 0
    return {"bracket_logprob": float(torch.log_softmax(logits, dim=-1)[close].item())}


def simple_addition_probe(model, tokenizer, device: str) -> dict:
    ids = tokenizer.encode("Q: 2+3= A:")
    out = model(torch.tensor([ids], device=device))
    pred = out["logits"][0, -1].argmax().item()
    return {"simple_addition_next_token": int(pred)}

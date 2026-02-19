from __future__ import annotations

import torch


def run_probes(model, tokenizer, device: str) -> dict:
    model.eval()
    probes = {
        'repeat_after_me': _repeat_probe(model, tokenizer, device),
        'bracket_matching': _bracket_probe(model, tokenizer, device),
        'simple_addition': _addition_probe(model, tokenizer, device),
    }
    return probes


def _score_next(model, tokenizer, prompt: str, target: str, device: str) -> float:
    ids = tokenizer.encode(prompt + target)
    pids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.zeros_like(x, dtype=torch.float32)
    mask[:, len(pids)-1:] = 1
    return float(model.sequence_logprob(x, mask).item())


def _repeat_probe(model, tok, d):
    return _score_next(model, tok, 'say: abcabc', ' abcabc', d)


def _bracket_probe(model, tok, d):
    good = _score_next(model, tok, 'open: (([', ']))', d)
    bad = _score_next(model, tok, 'open: (([', '[[[', d)
    return 1.0 if good > bad else 0.0


def _addition_probe(model, tok, d):
    return _score_next(model, tok, 'Q: 2+3=\nA:', ' 5', d)

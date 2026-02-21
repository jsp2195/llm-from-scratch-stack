from __future__ import annotations

import re
import torch


# -------------------------------------------------
# Helpers
# -------------------------------------------------

pat_shorthand = re.compile(r"^\s*(\d+)\s*([+\-*/])\s*(\d+)\s*\??\s*$")


def get_pad_id(tok):
    for attr in ("pad_id", "pad_token_id"):
        if hasattr(tok, attr):
            pid = getattr(tok, attr)
            if pid is not None:
                return int(pid)

    if hasattr(tok, "special_to_id") and isinstance(tok.special_to_id, dict):
        if "<pad>" in tok.special_to_id:
            return int(tok.special_to_id["<pad>"])

    if hasattr(tok, "token_to_id"):
        pid = tok.token_to_id("<pad>")
        if pid is not None:
            return int(pid)

    if hasattr(tok, "eos_id"):
        return int(tok.eos_id)

    return 0


def qfmt(q: str) -> str:
    return f"Q: {q.strip()}\n### Response:\n"


def ensure_prompt_format(p: str) -> str:
    """
    Guarantees canonical SFT prompt structure.
    """
    s = (p or "").strip()

    if "### Response:" in s:
        return s if s.endswith("\n") else s + "\n"

    m = pat_shorthand.fullmatch(s)
    if m:
        a, op, b = m.group(1), m.group(2), m.group(3)
        return qfmt(f"What is {a} {op} {b}?")

    return qfmt(s)


def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# -------------------------------------------------
# Robust SFT step (response-only loss)
# -------------------------------------------------

def sft_step(model, tokenizer, batch, separator: str, device: str):
    """
    Batched response-only SFT with:
        • prompt normalization
        • BOS/EOS
        • safe truncation
        • padding
        • loss mask
    """

    pad_id = get_pad_id(tokenizer)
    bos_id = int(tokenizer.bos_id)
    eos_id = int(tokenizer.eos_id)

    max_ctx = int(model.cfg.max_seq_len)

    ids_list = []
    labels_list = []

    for ex in batch:

        prompt = ensure_prompt_format(ex["prompt"])
        resp = clean_text(ex["response"])

        full = prompt + separator + resp

        ids_full = [bos_id] + tokenizer.encode(full) + [eos_id]
        ids_cut = [bos_id] + tokenizer.encode(prompt + separator)
        cut = len(ids_cut)

        # ---------- safe truncation ----------
        if len(ids_full) > max_ctx:
            overflow = len(ids_full) - max_ctx

            # keep BOS at position 0
            ids_full = [bos_id] + ids_full[1 + overflow:]

            cut = max(1, cut - overflow)

        labels = ids_full.copy()
        labels[:cut] = [-100] * cut

        ids_list.append(ids_full)
        labels_list.append(labels)

    # ---------- batching ----------
    B = len(ids_list)
    max_len = max(len(x) for x in ids_list)

    x = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((B, max_len), -100, dtype=torch.long, device=device)

    for i, (ids, lab) in enumerate(zip(ids_list, labels_list)):
        L = len(ids)
        x[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        y[i, :L] = torch.tensor(lab, dtype=torch.long, device=device)

    loss_mask = ((y != -100) & (x != pad_id)).to(torch.float32)

    _, loss = model(x, labels=y, loss_mask=loss_mask)

    return loss

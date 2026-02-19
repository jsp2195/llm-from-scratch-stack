#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt["model"], out / "model_state_dict.pt")
    manifest = {"tokenizer_dir": args.tokenizer_dir, "config": ckpt.get("config", {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported to {out}")

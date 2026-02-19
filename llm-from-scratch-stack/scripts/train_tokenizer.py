#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from tokenizers import ByteLevelBPETokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg):
    out_dir = Path(cfg.get("out_dir", "artifacts/tokenizer_toy"))
    out_dir.mkdir(parents=True, exist_ok=True)
    texts_file = out_dir / "tokenizer_corpus.txt"
    src_path = Path(cfg.data.train_path)
    with src_path.open(encoding="utf-8") as f, texts_file.open("w", encoding="utf-8") as w:
        for line in f:
            if cfg.data.format == "jsonl":
                w.write(json.loads(line)[cfg.data.text_field] + "\n")
            else:
                w.write(line)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(texts_file)],
        vocab_size=2048,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.save_model(str(out_dir))
    tokenizer.save(str(out_dir / "tokenizer.json"))
    OmegaConf.save(cfg, out_dir / "tokenizer_train_config.yaml")
    print(f"Tokenizer saved at {out_dir}")


if __name__ == "__main__":
    main()

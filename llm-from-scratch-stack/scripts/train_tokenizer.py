#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@hydra.main(version_base=None, config_path='../configs', config_name='pretrain')
def main(cfg: DictConfig):
    out = Path(cfg.get('out_dir', 'artifacts/tokenizer_toy'))
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=int(cfg.model.vocab_size), special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
    tok.train([str(Path(hydra.utils.get_original_cwd()) / cfg.data.train_path)], trainer)
    tok.save(str(out / 'tokenizer.json'))
    (out / 'manifest.json').write_text(json.dumps({'vocab_size': tok.get_vocab_size()}))


if __name__ == '__main__':
    main()

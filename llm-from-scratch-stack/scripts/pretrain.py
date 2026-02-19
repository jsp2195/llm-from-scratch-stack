from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from llmstack.data.datamodule import DataModule
from llmstack.model.gpt import GPTModel
from llmstack.optim.adamw import build_adamw
from llmstack.optim.lion import Lion
from llmstack.optim.schedulers import build_scheduler
from llmstack.tokenization.tokenizer import Tokenizer
from llmstack.train.engine import Trainer
from llmstack.utils.logging import RunLogger
from llmstack.utils.manifest import write_manifest
from llmstack.utils.seed import seed_everything


def setup_run(cfg):
    run = datetime.now().strftime('%Y%m%d_%H%M%S_') + cfg.train.run_name
    run_dir = Path(cfg.train.out_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(run_dir)
    (run_dir / 'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    write_manifest(str(run_dir), OmegaConf.to_container(cfg, resolve=True), sys.argv)
    return run_dir

@hydra.main(version_base=None, config_path='../configs', config_name='pretrain')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    seed_everything(int(cfg.train.seed), bool(cfg.train.deterministic))
    run_dir = setup_run(cfg)
    tok = Tokenizer(cfg.tokenizer_path)
    cfg.model.vocab_size = tok.vocab_size
    dm = DataModule(cfg, tok)
    model = GPTModel(cfg.model).to(cfg.train.device)
    if cfg.train.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    opt = build_adamw(model, cfg.optim) if cfg.optim.name == 'adamw' else Lion(model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)
    sch = build_scheduler(opt, cfg.sched, cfg.train.max_steps)
    logger = RunLogger(str(run_dir), cfg.log.jsonl, cfg.log.tensorboard)
    trainer = Trainer(cfg, model, opt, sch, logger, dm.train_dataloader(), dm.val_dataloader(), cfg.train.device)
    if cfg.train.resume_path:
        trainer.load_checkpoint(cfg.train.resume_path)
    trainer.fit()
    logger.close()


if __name__ == '__main__':
    main()

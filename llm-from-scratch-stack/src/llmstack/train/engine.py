"""Core training engine."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from llmstack.eval.perplexity import evaluate_perplexity
from llmstack.train.amp import autocast_context, build_scaler
from llmstack.train.checkpointing import (
    build_train_state,
    load_checkpoint,
    restore_train_state,
    save_checkpoint,
)
from llmstack.train.metrics import TokensPerSecond
from llmstack.utils.distributed import is_rank0


class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler, train_loader, val_loader, logger, device: str):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.scaler = build_scaler(cfg.train.precision)
        self.step = 0
        self.best_val = float("inf")

    def save_checkpoint(self, path: str):
        state = build_train_state(
            self.model, self.optimizer, self.scheduler, self.scaler, self.step, self.best_val, self.cfg
        )
        save_checkpoint(path, state)

    def load_checkpoint(self, path: str):
        ckpt = load_checkpoint(path, map_location=self.device)
        self.step, self.best_val = restore_train_state(
            self.model, self.optimizer, self.scheduler, self.scaler, ckpt
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        return evaluate_perplexity(self.model, self.val_loader, self.device, self.cfg.train.max_eval_batches)[
            "val_loss"
        ]

    def fit(self):
        if self.cfg.train.resume_path:
            self.load_checkpoint(self.cfg.train.resume_path)
        meter = TokensPerSecond()
        self.model.train()
        loop = tqdm(total=self.cfg.train.max_steps, disable=not is_rank0())
        while self.step < self.cfg.train.max_steps:
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                with autocast_context(self.device, self.cfg.train.precision):
                    out = self.model(input_ids, labels)
                    loss = out["loss"] / self.cfg.train.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (self.step + 1) % self.cfg.train.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler:
                        self.scheduler.step()
                self.step += 1
                if self.step % self.cfg.train.log_interval == 0 and is_rank0():
                    tok = input_ids.numel() * self.cfg.train.log_interval
                    self.logger.log(
                        self.step,
                        {
                            "train_loss": float(loss.item() * self.cfg.train.grad_accum_steps),
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "tokens_per_sec": meter.compute(tok),
                        },
                    )
                if self.step % self.cfg.train.eval_interval == 0:
                    val_loss = self.evaluate()
                    if is_rank0():
                        ppl = float(torch.exp(torch.tensor(val_loss)))
                        self.logger.log(self.step, {"val_loss": val_loss, "ppl": ppl})
                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self.save_checkpoint(str(Path(self.cfg.train.run_dir) / "best.pt"))
                    self.model.train()
                if self.step % self.cfg.train.save_interval == 0:
                    self.save_checkpoint(str(Path(self.cfg.train.run_dir) / "last.pt"))
                loop.update(1)
                if self.step >= self.cfg.train.max_steps:
                    break
        self.save_checkpoint(str(Path(self.cfg.train.run_dir) / "last.pt"))
        loop.close()

from __future__ import annotations

import time
from contextlib import nullcontext
from tqdm.auto import tqdm
import torch

from llmstack.eval.perplexity import evaluate_perplexity
from llmstack.train.amp import autocast_context, make_scaler
from llmstack.train.checkpointing import capture_rng, restore_rng, save_checkpoint
from llmstack.utils.distributed import is_rank0


class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler, logger, train_loader, val_loader, device: str = 'cpu'):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = make_scaler(cfg.train.precision, device)
        self.step = 0
        self.best_val = float('inf')

    def fit(self):
        self.model.train()
        t0 = time.time()
        loader_iter = iter(self.train_loader)

        progress = tqdm(total=self.cfg.train.max_steps, desc="Pretraining") if is_rank0() else None  # NEW

        while self.step < self.cfg.train.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            acc_loss = 0.0

            for micro in range(self.cfg.train.grad_accum_steps):
                batch = next(loader_iter, None)
                if batch is None:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch['loss_mask'].to(self.device)

                sync_ctx = self.model.no_sync if hasattr(self.model, 'no_sync') and micro < self.cfg.train.grad_accum_steps - 1 else nullcontext

                with sync_ctx():
                    with autocast_context(self.cfg.train.precision, self.device):
                        _, loss = self.model(ids, labels=labels, loss_mask=mask)
                        loss = loss / self.cfg.train.grad_accum_steps

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                acc_loss += float(loss.item())

            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.step += 1

            if progress is not None:  # NEW
                progress.update(1)     # NEW

            if self.step % self.cfg.train.log_interval == 0 and is_rank0():
                dt = time.time() - t0
                tokens_per_step = (
                    self.cfg.train.micro_batch_size *
                    self.cfg.train.grad_accum_steps *
                    self.cfg.data.seq_len
                )
                tokens = self.step * tokens_per_step
                
                self.logger.log(self.step, {
                    'loss': acc_loss,
                    'lr': self.scheduler.get_last_lr()[0],
                    'grad_norm': float(grad_norm),
                    'tokens_per_sec': tokens / max(dt, 1e-6)
                })

            if self.step % self.cfg.train.eval_interval == 0:
                val = self.evaluate()
                if is_rank0() and val < self.best_val:
                    self.best_val = val
                    self.save_checkpoint('best.ckpt')

            if self.step % self.cfg.train.save_interval == 0 and is_rank0():
                self.save_checkpoint('last.ckpt')

        if progress is not None:  # NEW
            progress.close()      # NEW

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val = evaluate_perplexity(self.model, self.val_loader, self.device, self.cfg.train.max_eval_batches)['loss']
        self.model.train()
        if is_rank0():
            self.logger.log(self.step, {'val_loss': val})
        return val

    def save_checkpoint(self, name: str):
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'scaler': self.scaler.state_dict() if self.scaler else None, 'step': self.step, 'rng': capture_rng(), 'config': self.cfg}
        save_checkpoint(f"{self.cfg.run_dir}/{name}", state)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        if self.scaler and ckpt.get('scaler'):
            self.scaler.load_state_dict(ckpt['scaler'])
        self.step = ckpt['step']
        restore_rng(ckpt['rng'])

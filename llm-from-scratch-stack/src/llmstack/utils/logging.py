"""JSONL + TensorBoard logging."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from llmstack.utils.distributed import is_rank0


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.tb = SummaryWriter(str(self.run_dir / "tb")) if is_rank0() else None

    def log(self, step: int, metrics: dict):
        if not is_rank0():
            return
        item = {"step": step, **metrics}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self):
        if self.tb:
            self.tb.close()

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, out_dir: str, jsonl: bool = True, tensorboard: bool = True):
        self.out_dir = Path(out_dir)
        self.jsonl = self.out_dir / 'metrics.jsonl' if jsonl else None
        self.tb = SummaryWriter(self.out_dir / 'tb') if tensorboard else None

    def log(self, step: int, metrics: dict):
        if self.jsonl:
            with self.jsonl.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'step': step, **metrics}) + '\\n')
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self):
        if self.tb:
            self.tb.close()

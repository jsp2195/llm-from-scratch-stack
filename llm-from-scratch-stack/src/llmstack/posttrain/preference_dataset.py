from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

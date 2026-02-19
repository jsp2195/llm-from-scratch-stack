"""Run manifest writer."""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(run_dir: str, cfg, argv: list[str]) -> None:
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
        "git_sha": _git_sha(),
        "command": argv,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    p = Path(run_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

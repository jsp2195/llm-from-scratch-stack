from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path


def _cmd_out(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def write_manifest(out_dir: str, cfg, argv: list[str]) -> None:
    p = Path(out_dir)
    man = {
        'timestamp': datetime.utcnow().isoformat(),
        'hostname': socket.gethostname(),
        'git_sha': _cmd_out(['git', 'rev-parse', 'HEAD']),
        'pip_freeze': _cmd_out(['pip', 'freeze']).splitlines(),
        'cmdline': argv,
        'config': cfg,
    }
    (p / 'manifest.json').write_text(json.dumps(man, indent=2))

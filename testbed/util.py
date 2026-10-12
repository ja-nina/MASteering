"""Small shared helpers: logging, atomic file writes, deterministic seeds."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def log(msg: str) -> None:
    # flush so SLURM .out files show progress live
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic(path, write) -> None:
    """Write to a per-process temp file, then rename: a killed job never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def write_json(path, data) -> None:
    _atomic(path, lambda p: p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pt(path, obj) -> None:
    import torch
    _atomic(path, lambda p: torch.save(obj, p))


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from any parts, e.g. stable_seed(run_id, episode, "steer")."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) % 2**31

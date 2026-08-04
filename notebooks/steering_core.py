from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    trait: str
    layer: int
    start: int
    end: Optional[int]   # None = steer until end of generation
    coeff: float
    mode: str = "additive"  # "additive" | "adaptive"


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def load_vectors(vectors_dir: str) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load all *.pt trait vectors from a PersVecGen bf16 directory.

    Returns {trait_name: {layer_int: tensor}}.
    Plain-tensor files (not dicts) are stored as {-1: tensor}.
    """
    result: Dict[str, Dict[int, torch.Tensor]] = {}
    for pt_path in sorted(pathlib.Path(vectors_dir).glob("*.pt")):
        trait = pt_path.stem
        loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            result[trait] = {int(k): v.float() for k, v in loaded.items()}
        else:
            result[trait] = {-1: loaded.float()}
    return result


def best_layer(trait: str, vectors_dir: str) -> Optional[int]:
    """Return the integer layer with the highest sweep delta for a trait.

    Reads the companion <trait>.json file (PersVecGen metadata).
    Returns None if the file does not exist.
    """
    json_path = pathlib.Path(vectors_dir) / f"{trait}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        meta = json.load(f)
    sweep = meta.get("sweep_scores", {})
    if not sweep:
        return None

    def _delta(lk: str) -> float:
        scores = {float(k): float(v) for k, v in sweep[lk].items()}
        baseline = scores.get(0.0, min(scores.values()))
        return max(scores.values()) - baseline

    return int(max(sweep.keys(), key=_delta))

"""Norms of the persona vectors per layer, for the report's push-size analysis (section 26).

    python scripts/vector_norms.py --sweep config/sweeps/focus.yaml [--model qwen3.8-27b] --out results/<sweep>/vector_norms.json

A steered agent's hidden state becomes h + c·v at one block: the push is c times the norm of v. The random control
(testbed/steering.py, random_control) has the norm of the trait's vector at that layer, so the same table gives the
size of every random push too; where PersVecFast exported its own random-direction file (vllm_hook/<trait>_L<l>
_random.pt), that file's norm is recorded instead ("random"). Writes {"model", "vectors_dir", "norms": {trait:
{layer: |v|}}, "random": {trait: {layer: |v|}}}; scripts/report.py reads it from next to the agents CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def norms(vectors_dir: Path) -> dict:
    import torch
    from testbed.steering import VectorStore, slugify
    store = VectorStore(vectors_dir, cache_dir=vectors_dir)  # only reads
    out, rand = {}, {}
    for t in store.traits():
        out[t] = {int(l): float(v.norm()) for l, v in store.vectors(t).items()}
        for l in out[t]:
            exported = vectors_dir / "vllm_hook" / f"{slugify(t)}_L{l}_random.pt"
            if exported.exists():
                rand.setdefault(t, {})[l] = float(torch.load(exported, map_location="cpu")["dir"].float().norm())
    return {"norms": out, "random": rand}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--model", help="the model key the sweep was run with (make_configs --model), if not its own")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    from scripts.make_configs import load_sweep
    from testbed.config import DEFAULT_PVF_OUT
    sweep = load_sweep(args.sweep, args.model)
    model = sweep["model"]
    vectors_dir = Path(os.path.expandvars(sweep.get("vectors_dir") or
                                          f"{os.environ.get('PVF_OUT', DEFAULT_PVF_OUT)}/persona/{model}/bf16"))
    data = {"model": model, "vectors_dir": str(vectors_dir), **norms(vectors_dir)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{len(data['norms'])} trait(s) x {len(next(iter(data['norms'].values()), {}))} layer(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

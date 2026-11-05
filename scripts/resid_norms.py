"""Residual-stream norms per layer, from the activations PersVecFast saved when it extracted the persona vectors
(_stage/<trait>/acts_<model>.pt: per answer, the residual stream after each block averaged over the generated
tokens, [n_layers, hidden]). With vector_norms.json this gives the steering strength relative to the hidden state,
||c v_l|| / ||h_l||, without a GPU. Caveat: the norm of a token-averaged residual is below the typical per-token
norm, so the ratio is an upper bound on the per-token one.

    python scripts/resid_norms.py $PVF_OUT/persona --out results/resid_norms.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def norms(path: Path) -> dict:
    acts = torch.load(path, map_location="cpu")
    rows = [a.float() for side in ("positive", "negative") for a in acts.get(side, []) if a is not None]
    if not rows:
        return {}
    x = torch.stack(rows)                                   # [n_answers, n_layers, hidden]
    per = x.norm(dim=-1)                                    # [n_answers, n_layers]
    return {str(l): {"median": round(float(per[:, l].median()), 3), "n": int(per.shape[0])}
            for l in range(per.shape[1])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="searched recursively for _stage/<trait>/acts_<model>.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out: dict = {}
    for p in sorted(Path(a.root).rglob("acts_*.pt")):
        if p.parent.parent.name != "_stage":
            continue
        model = p.stem[len("acts_"):]
        res = norms(p)
        if res:
            out.setdefault(model, {})[p.parent.name] = res
            print(f"{model} {p.parent.name}: {len(res)} layers, n={next(iter(res.values()))['n']}", flush=True)
    # per model and layer: median over traits (the hidden state does not depend much on the trait's prompt)
    summary = {m: {l: round(statistics.median(t[l]["median"] for t in traits.values() if l in t), 3)
                   for l in next(iter(traits.values()))} for m, traits in out.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"by_model_layer": summary, "by_trait": out}, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

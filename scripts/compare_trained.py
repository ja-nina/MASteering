"""Trained vs source persona vectors in the games: the same measures for each source trait <t> and its trained twin
<t>-<suffix> (scripts/export_trained_vectors.py), played in one sweep with the same seeds, plus the same measures on
another sweep's source vectors (e.g. the write-up's main_v2) for reference.

    python scripts/compare_trained.py results/trained_games/agents.csv.gz --suffix kl10t2048 \\
        --ref results/main_v2/agents.csv.gz --max-c 1 --out results/trained_games/compare_trained.json

Measures (steered agents of the vector conditions one / half / all unless said; |c| <= --max-c; readouts in SD units
of the unsteered agents of the same game cell = group x observation mode; slopes per unit c with game-cell fixed
effects and SEs clustered by episode, scripts/report.py's slope):
  on_target     readout of the SOURCE trait direction <t> (the same ruler for both vectors), vs c
  on_own        the trained twin read on its own direction <t>-<suffix>
  random        all_random (norm-matched random direction) read on <t>
  off_target    each other source trait's readout vs c; off_mean_abs = mean |slope| over them, off_ratio = that / on_target
  contagion     condition one: the unsteered co-players' readout of <t>
  reward_one    condition one: the steered agent's paired reward change (same seat, same seed, unsteered episode),
                in SD units of the unsteered rewards of that game cell
  welfare_all   condition all: the group's paired reward change, in SD units of the unsteered group sums
  fallbacks     parse fallbacks per turn of steered agents at |c| = max c, and of unsteered agents
  per_game      on_target and reward_one per game cell
Reads only the rows it needs (conditions none, one, half, all, all_random), so a large sweep fits in memory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import report as R  # noqa: E402

CONDITIONS = ("none", "one", "half", "all", "all_random")
VECTOR = ("one", "half", "all")


def load_rows(path, traits, max_c):
    """Compact agent rows of `path`: the conditions above, |c| <= max_c, readouts of `traits` only."""
    rows = []
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
    for r in R.read_csv(path):
        cond = r.get("condition") or "none"
        if cond not in CONDITIONS:
            continue
        coef = num(r, "condition_coefficient")
        if coef is not None and abs(coef) > max_c + 1e-9:
            continue
        if r.get("condition_via") not in (None, "", "vector"):
            continue
        group, obs = r["group"], r.get("observation") or "text"
        rows.append({
            "group": group, "obs": obs, "cell": (group, obs), "condition": cond,
            "cluster": (r["run_id"], int(r["episode"])), "episode": int(r["episode"]), "agent": r["agent"],
            "trait": r.get("condition_trait") or None, "coef": coef, "steered": r.get("steered") == "True",
            "reward": num(r, "reward"), "score": num(r, "score"),
            "turns": num(r, "turns") or 0, "fallbacks": num(r, "parse_fallbacks") or 0,
            "mon": {t: float(r[f"mon_{t}"]) for t in traits if r.get(f"mon_{t}") not in (None, "")}})
    R.use_scores(rows)
    return rows


def zscore(rows):
    vals = defaultdict(list)
    for r in rows:
        if r["condition"] == "none":
            for t, v in r["mon"].items():
                vals[(r["cell"], t)].append(v)
    st = {k: (float(np.mean(v)), float(np.std(v)) or 1.0) for k, v in vals.items() if len(v) > 1}
    for r in rows:
        r["z"] = {t: (v - st[(r["cell"], t)][0]) / st[(r["cell"], t)][1] for t, v in r["mon"].items()
                  if (r["cell"], t) in st}


def _slope(pts, y):
    pts = [r for r in pts if y(r) is not None]
    s = R.slope([r["coef"] if r["condition"] != "none" else 0.0 for r in pts], [y(r) for r in pts],
                [r["cluster"] for r in pts], [r["cell"] for r in pts])
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in s.items() if k != "games"}


def readout(rows, base_by_cell, t, u, select):
    """Slope of readout u over c for the rows `select` picks (trait t), unsteered agents of those cells at c = 0."""
    chosen = [r for r in rows if r["trait"] == t and select(r) and u in r["z"]]
    cells = {r["cell"] for r in chosen}
    pts = chosen + [r for c in cells for r in base_by_cell.get(c, []) if u in r["z"]]
    return _slope(pts, lambda r: r["z"][u])


def measures(rows, t, sources, max_c, own=None):
    """All measures for the vector of trait t (rows of one sweep); readouts on source trait `own or t`'s ruler."""
    src = own or t
    base_by_cell = defaultdict(list)
    for r in rows:
        if r["condition"] == "none":
            base_by_cell[r["cell"]].append(r)
    vec = lambda r: r["steered"] and r["condition"] in VECTOR
    out = {"on_target": readout(rows, base_by_cell, t, src, vec)}
    if own:
        out["on_own"] = readout(rows, base_by_cell, t, t, vec)
    out["random"] = readout(rows, base_by_cell, t, src, lambda r: r["steered"] and r["condition"] == "all_random")
    off = {u: readout(rows, base_by_cell, t, u, vec)["b"] for u in sources if u != src}
    off = {u: b for u, b in off.items() if b is not None}
    out["off_target"] = off
    out["off_mean_abs"] = float(np.mean([abs(b) for b in off.values()])) if off else None
    on = out["on_target"]["b"]
    out["off_ratio"] = out["off_mean_abs"] / abs(on) if on and out["off_mean_abs"] is not None else None
    out["contagion"] = readout(rows, base_by_cell, t, src, lambda r: not r["steered"] and r["condition"] == "one")

    base = R.baselines(rows)
    scale = R.reward_scale(rows)
    pr = [p for p in R.paired(rows, base, lambda r: r["trait"] == t and r["steered"] and r["condition"] == "one")]
    for p in pr:
        p["diff"] /= scale.get(p["cell"], 1.0)
    cells = {p["cell"] for p in pr}
    out["reward_one"] = _slope(pr, lambda r: r["diff"])
    wf = R.welfare([r for r in rows if r["trait"] == t and r["condition"] == "all"],
                   [r for r in rows if r["condition"] == "none"])
    out["welfare_all"] = _slope(wf, lambda r: r["diff"])

    def fb(sel):
        rs = [r for r in rows if sel(r) and r["turns"]]
        return sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs)) if rs else None
    out["fallbacks"] = {"steered_maxc": fb(lambda r: vec(r) and r["trait"] == t and abs(r["coef"] or 0) >= max_c - 1e-9),
                        "unsteered": fb(lambda r: r["condition"] == "none")}
    out["per_game"] = {}
    for cell in sorted({r["cell"] for r in rows if r["trait"] == t and vec(r)}):
        sub = [r for r in rows if r["cell"] == cell]
        bc = {cell: base_by_cell.get(cell, [])}
        rw = [p for p in pr if p["cell"] == cell]
        out["per_game"][f"{cell[0]}/{cell[1]}"] = {
            "on_target": readout(sub, bc, t, src, vec)["b"],
            "reward_one": _slope(rw, lambda r: r["diff"])["b"] if cell in cells else None}
    return out


def fmt(s, d=2):
    if not s or s.get("b") is None:
        return "–"
    se = s.get("se")
    return f"{s['b']:+.{d}f}" + (f" ± {se:.{d}f}" if se else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="agents.csv.gz of the sweep with the source traits and their trained twins")
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--traits", nargs="*", default=["evil", "sarcastic", "empathy", "assertive", "sycophantic",
                                                    "opportunistic", "agreeableness"])
    ap.add_argument("--ref", help="another sweep's agents.csv.gz (source vectors only), e.g. the write-up's main_v2")
    ap.add_argument("--max-c", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = a.traits
    twins = {t: f"{t}-{a.suffix}" for t in src}
    rows = load_rows(a.csv, src + list(twins.values()), a.max_c)
    zscore(rows)
    groups = sorted({r["group"] for r in rows})
    print(f"{a.csv}: {len(rows)} agent rows, {len(groups)} groups: {', '.join(groups)}")
    res = {"csv": a.csv, "suffix": a.suffix, "max_c": a.max_c, "groups": groups, "traits": {}}
    for t in src:
        res["traits"][t] = {"source": measures(rows, t, src, a.max_c),
                            "trained": measures(rows, twins[t], src, a.max_c, own=t)}
    if a.ref:
        ref = load_rows(a.ref, src, a.max_c)
        ref = [r for r in ref if r["group"] in groups]  # the games both sweeps played
        zscore(ref)
        res["ref"] = {"csv": a.ref, "groups": sorted({r["group"] for r in ref})}
        for t in src:
            res["traits"][t]["ref"] = measures(ref, t, src, a.max_c)
        del ref
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    cols = ["source", "trained"] + (["ref"] if a.ref else [])
    for key, title in (("on_target", "on-target readout slope (SD per unit c)"), ("random", "random control"),
                       ("contagion", "contagion (co-players)"), ("reward_one", "reward, one steered (SD)"),
                       ("welfare_all", "welfare, all steered (SD)")):
        print(f"\n{title}: " + " | ".join(cols))
        for t in src:
            print(f"  {t:14} " + " | ".join(fmt(res['traits'][t][c][key]) for c in cols))
    print("\noff-target mean |slope| (ratio to on-target): " + " | ".join(cols))
    for t in src:
        print(f"  {t:14} " + " | ".join(
            f"{res['traits'][t][c]['off_mean_abs']:.2f} ({res['traits'][t][c]['off_ratio']:.2f})"
            if res['traits'][t][c]['off_ratio'] is not None else "–" for c in cols))
    print("\nparse fallbacks per turn at |c| = max (unsteered): " + " | ".join(cols))
    for t in src:
        print(f"  {t:14} " + " | ".join(
            f"{res['traits'][t][c]['fallbacks']['steered_maxc'] or 0:.3f} ({res['traits'][t][c]['fallbacks']['unsteered'] or 0:.3f})"
            for c in cols))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

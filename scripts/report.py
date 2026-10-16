"""Analysis report over every game of a sweep: one self-contained HTML page.

    python scripts/summarize_runs.py <logs>/main [fallback/main] --out results/main/agents.csv.gz
    python scripts/report.py results/main/agents.csv.gz --sweep config/sweeps/main.yaml \n        --logs <logging_dir>/main --out results/main/report.html

Reads the agent-episode rows of summarize_runs.py and answers, per trait and per game:
   1 coverage        episodes per game, truncated episodes, unparseable answers
   2 manipulation    does steering move the steered agent's trait readout? (dose-response, random control,
                     persona prompt) The readout: the agent's answers read by the UNsteered model, its hidden
                     state at the monitor layer projected on the trait's unit vector, in SD units of the
                     unsteered agents of the same game
   3 specificity     steering one trait: which other traits' readouts move (trait x trait)
   4 detectability   can a monitor single out the steered agent among its co-players (AUC by coefficient)
   5 asymmetry       amplifying (c > 0) vs suppressing (c < 0)
   6 channels        persona prompt vs vector vs both
   7 who is steered  one / half / all / evil team / good one ...: readout, own reward, group welfare
   8 contagion       the unsteered co-players' readout of the steered trait
   9-11 outcomes     paired reward effects (the same seat in the unsteered episode with the same seed):
                     steered agent per game, spillover on the others, the evil team
  11b link          does the trait in the language go with the outcome? (readout -> reward, and how
                     much of the steering effect on reward the readout accounts for)
  12 text vs image   the same games with image observations
  13 coherence       unparseable answers and answer length vs coefficient
Slopes come with standard errors clustered by episode; p-values (normal approximation) are
Benjamini-Hochberg corrected over all cells of a table (q). Pooled slopes across games use game
fixed effects; for rewards, each game's rewards are first scaled by its unsteered reward SD.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

STEERED_VIA = ("vector", "both")


# ---------------------------------------------------------------- data

def load(path) -> list:
    rows = []
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            num = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
            row = {
                "run_id": r["run_id"], "group": r.get("group") or r["run_id"].split("__")[0],
                "condition": r.get("condition") or (r["run_id"].split("__")[1] if "__" in r["run_id"] else "none"),
                "obs": r.get("observation") or "text", "episode": int(r["episode"]), "agent": r["agent"],
                "trait": r.get("condition_trait") or None, "coef": num("condition_coefficient"),
                "via": r.get("condition_via") or None, "random": r.get("condition_random") == "True",
                "team_cond": r.get("condition_team") or None, "select": r.get("condition_select") or None,
                "steered": r.get("steered") == "True", "team": r.get("team") or None,
                "reward": num("reward"), "turns": num("turns") or 0, "fallbacks": num("parse_fallbacks") or 0,
                "truncated": r.get("truncated") == "True", "chars": num("answer_chars"),
                "monitor_layer": num("monitor_layer"), "steer_layer": num("steer_layer"),
                "mon": {k[4:]: float(v) for k, v in r.items() if k.startswith("mon_") and v not in (None, "")},
            }
            rows.append(row)
    for r in rows:
        r["cell"] = (r["group"], r["obs"])  # a game as played: group and observation mode
        r["cluster"] = (r["run_id"], r["episode"])
    return rows


def baselines(rows) -> dict:
    """{(group, obs, episode, agent): reward} of the unsteered runs: same seed = same deal and roles."""
    return {(r["group"], r["obs"], r["episode"], r["agent"]): r["reward"]
            for r in rows if r["condition"] == "none" and r["reward"] is not None}


# ---------------------------------------------------------------- statistics

def slope(x, y, clusters, fe=None) -> dict:
    """OLS slope of y on x (with fixed effects `fe` if given), SE clustered by `clusters`."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    clusters = [c for c, k in zip(clusters, keep) if k]
    n = len(x)
    if n < 3:
        return {"n": n, "b": float("nan"), "se": float("nan"), "p": float("nan")}
    if fe is not None:
        fe = [f for f, k in zip(fe, keep) if k]
        x, y = _demean(x, fe), _demean(y, fe)
    else:
        x, y = x - x.mean(), y - y.mean()
    sxx = float((x * x).sum())
    if sxx <= 0:
        return {"n": n, "b": float("nan"), "se": float("nan"), "p": float("nan")}
    b = float((x * y).sum() / sxx)
    resid = y - b * x
    ids = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    g = np.array([ids[c] for c in clusters])
    scores = np.bincount(g, weights=x * resid)
    k = len(ids)
    var = float((scores ** 2).sum()) / sxx ** 2 * (k / (k - 1) if k > 1 else 1.0)
    # a perfect fit (e.g. every paired difference 0) leaves only rounding noise: no inference
    se = math.sqrt(var) if var > 0 and np.abs(resid).max() > 1e-9 else float("nan")
    p = math.erfc(abs(b / se) / math.sqrt(2)) if se and math.isfinite(se) and se > 0 else float("nan")
    return {"n": n, "clusters": k, "b": b, "se": se, "p": p}


def _demean(v, groups):
    groups = list(groups)
    ids = {gr: i for i, gr in enumerate(dict.fromkeys(groups))}
    g = np.array([ids[gr] for gr in groups])
    means = np.bincount(g, weights=v) / np.bincount(g)
    return v - means[g]


def mean_ci(values, clusters):
    """Mean with a 95% CI from the cluster means' spread."""
    by = defaultdict(list)
    for v, c in zip(values, clusters):
        if v is not None and math.isfinite(v):
            by[c].append(v)
    cm = np.array([np.mean(v) for v in by.values()])
    if len(cm) == 0:
        return float("nan"), float("nan"), 0
    m = float(np.mean([v for vs in by.values() for v in vs]))
    half = 1.96 * float(cm.std(ddof=1)) / math.sqrt(len(cm)) if len(cm) > 1 else float("nan")
    return m, half, len(cm)


def bh(pvalues):
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    idx = [i for i, p in enumerate(pvalues) if p == p]
    q = [float("nan")] * len(pvalues)
    order = sorted(idx, key=lambda i: pvalues[i])
    m, prev = len(order), 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvalues[i] * m / rank)
        q[i] = prev
    return q


# ---------------------------------------------------------------- analyses

def manipulation(rows, traits):
    """Per trait: the steered agents' readout of their own trait vs the coefficient."""
    base_mon = defaultdict(list)  # unsteered agents' readout per game cell, as the coefficient-0 point
    for r in rows:
        if r["condition"] == "none":
            for t, v in r["mon"].items():
                base_mon[(r["cell"], t)].append(v)
    out = {}
    for t in traits:
        vec = [r for r in rows if r["steered"] and r["trait"] == t and r["via"] in STEERED_VIA and not r["random"]
               and t in r["mon"]]
        cells = {r["cell"] for r in vec}
        base = [r for r in rows if r["condition"] == "none" and r["cell"] in cells and t in r["mon"]]
        pts = vec + base
        x = [r["coef"] if r["steered"] else 0.0 for r in pts]
        res = {"vector": slope(x, [r["mon"][t] for r in pts], [r["cluster"] for r in pts], [r["cell"] for r in pts])}
        rnd = [r for r in rows if r["steered"] and r["trait"] == t and r["random"] and t in r["mon"]]
        rp = rnd + [r for r in base if r["cell"] in {q["cell"] for q in rnd}]
        res["random"] = slope([r["coef"] if r["steered"] else 0.0 for r in rp], [r["mon"][t] for r in rp],
                              [r["cluster"] for r in rp], [r["cell"] for r in rp])
        # dose-response curve, each readout minus its game cell's unsteered mean
        centre = {c: float(np.mean(base_mon[(c, t)])) for c in cells if base_mon.get((c, t))}
        curve = defaultdict(lambda: ([], []))
        for r in vec + base + rnd:
            if r["cell"] not in centre:
                continue
            key = ("random" if r["random"] else "vector", r["coef"] if r["steered"] else 0.0)
            curve[key][0].append(r["mon"][t] - centre[r["cell"]])
            curve[key][1].append(r["cluster"])
        res["curve"] = {k: mean_ci(*v) for k, v in curve.items()}
        prompt = [r for r in rows if r["steered"] and r["trait"] == t and r["via"] == "prompt" and t in r["mon"]
                  and r["cell"] in centre]
        res["prompt"] = {sign: mean_ci([r["mon"][t] - centre[r["cell"]] for r in prompt if (r["coef"] or 0) * sign > 0],
                                       [r["cluster"] for r in prompt if (r["coef"] or 0) * sign > 0])
                         for sign in (1, -1)}
        res["per_game"] = {}
        for cell in sorted(cells):
            cp = [r for r in pts if r["cell"] == cell]
            res["per_game"][cell] = slope([r["coef"] if r["steered"] else 0.0 for r in cp], [r["mon"][t] for r in cp],
                                          [r["cluster"] for r in cp])
        out[t] = res
    return out


def zscored(rows):
    """Readouts in SD units of the unsteered agents of the same game cell (0 = a typical unsteered agent)."""
    vals = defaultdict(list)
    for r in rows:
        if r["condition"] == "none":
            for t, v in r["mon"].items():
                vals[(r["cell"], t)].append(v)
    stats = {k: (float(np.mean(v)), float(np.std(v)) or 1.0) for k, v in vals.items() if len(v) > 1}
    out = []
    for r in rows:
        z = {t: (v - stats[(r["cell"], t)][0]) / stats[(r["cell"], t)][1]
             for t, v in r["mon"].items() if (r["cell"], t) in stats}
        out.append({**r, "mon_raw": r["mon"], "mon": z})
    return out


def is_vector(r, trait=None):
    """A steered agent that got the persona vector (alone or with the prompt), not the random control."""
    return r["steered"] and r["via"] in STEERED_VIA and not r["random"] and (trait is None or r["trait"] == trait)


def readout_slope(rows, t, select, readout=None, by_cell=True):
    """Slope of the `readout` trait's readout over the coefficient for the agents `select` picks, with the
    unsteered agents of the same game cells as the coefficient-0 points; game fixed effects."""
    u = readout or t
    chosen = [r for r in rows if r["trait"] == t and r["coef"] is not None and u in r["mon"] and select(r)]
    cells = {r["cell"] for r in chosen}
    base = [r for r in rows if r["condition"] == "none" and r["cell"] in cells and u in r["mon"]]
    pts = chosen + base
    return slope([r["coef"] if r["condition"] != "none" else 0.0 for r in pts], [r["mon"][u] for r in pts],
                 [r["cluster"] for r in pts], [r["cell"] for r in pts] if by_cell else None)


def welfare(rows, base_rows):
    """Per steered episode: the sum of all agents' rewards minus that of the unsteered episode with the same
    seed, in SD units of the unsteered episodes' sums of that game cell."""
    sums, info = defaultdict(float), {}
    for r in rows:
        if r["reward"] is not None:
            sums[r["cluster"]] += r["reward"]
            info[r["cluster"]] = r
    base_sum, base_by_cell = {}, defaultdict(list)
    for r in base_rows:
        if r["reward"] is not None:
            base_sum[(r["group"], r["obs"], r["episode"])] = base_sum.get((r["group"], r["obs"], r["episode"]), 0.0) \
                                                             + r["reward"]
    for (g, o, e), s in base_sum.items():
        base_by_cell[(g, o)].append(s)
    sd = {c: (float(np.std(v)) or 1.0) if len(v) > 1 else 1.0 for c, v in base_by_cell.items()}
    out = []
    for cl, s in sums.items():
        r = info[cl]
        if r["condition"] == "none":
            continue
        b = base_sum.get((r["group"], r["obs"], r["episode"]))
        if b is not None:
            out.append({**r, "diff": (s - b) / sd.get(r["cell"], 1.0)})
    return out


def detectability(rows, t, via=STEERED_VIA):
    """{coefficient: (mean AUC, half CI, episodes)}: how well the steered agent's own-trait readout singles it
    out among the co-players of its episode (single-agent conditions, steered `via` these channels).
    0.5 = invisible to the monitor; for negative coefficients the steered agent is expected to read LOWER,
    and the AUC is oriented so."""
    episodes = defaultdict(list)
    for r in rows:
        if r["trait"] == t and r["select"] == "one" and r["via"] in via and not r["random"] \
                and r["team_cond"] is None and t in r["mon"]:
            episodes[r["cluster"]].append(r)
    by_coef = defaultdict(lambda: ([], []))
    for cl, rs in episodes.items():
        st = [r for r in rs if r["steered"]]
        others = [r["mon"][t] for r in rs if not r["steered"]]
        if len(st) != 1 or not others:
            continue
        s, c = st[0]["mon"][t], st[0]["coef"]
        sign = 1 if c >= 0 else -1
        auc = float(np.mean([1.0 if sign * (s - o) > 0 else 0.5 if s == o else 0.0 for o in others]))
        by_coef[c][0].append(auc)
        by_coef[c][1].append(cl)
    return {c: mean_ci(*v) for c, v in sorted(by_coef.items())}


def ols(y, X, clusters, fe=None):
    """Multiple regression y ~ X (columns, with game fixed effects `fe`); coefficients with SEs clustered by
    `clusters`. Returns [{"b", "se", "p", "n"}] per column."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[keep], X[keep]
    clusters = [c for c, k in zip(clusters, keep) if k]
    nan = [{"b": float("nan"), "se": float("nan"), "p": float("nan"), "n": len(y)}] * X.shape[1]
    if len(y) <= X.shape[1] + 2:
        return nan
    if fe is not None:
        fe = [f for f, k in zip(fe, keep) if k]
        y, X = _demean(y, fe), np.column_stack([_demean(X[:, j], fe) for j in range(X.shape[1])])
    else:
        y, X = y - y.mean(), X - X.mean(axis=0)
    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) < X.shape[1]:
        return nan
    inv = np.linalg.inv(xtx)
    b = inv @ X.T @ y
    e = y - X @ b
    ids = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    g = np.array([ids[c] for c in clusters])
    scores = np.zeros((len(ids), X.shape[1]))
    np.add.at(scores, g, X * e[:, None])
    k = len(ids)
    cov = inv @ (scores.T @ scores) @ inv * (k / (k - 1) if k > 1 else 1.0)
    out = []
    for j in range(X.shape[1]):
        se = math.sqrt(cov[j, j]) if cov[j, j] > 0 and np.abs(e).max() > 1e-9 else float("nan")
        p = math.erfc(abs(b[j] / se) / math.sqrt(2)) if se == se and se > 0 else float("nan")
        out.append({"b": float(b[j]), "se": se, "p": p, "n": len(y), "clusters": k})
    return out


def link(steered, t, scale):
    """How the steered agents' readout relates to their outcome (paired reward difference, SD units):
    total effect of the coefficient on the outcome; association of the readout with the outcome; and
    the coefficient's effect once the readout is held fixed (what the readout does not account for)."""
    rs = [r for r in steered if r["trait"] == t and r["coef"] is not None and t in r["mon"]]
    y = [r["diff"] / scale.get(r["cell"], 1.0) for r in rs]
    cl, fe = [r["cluster"] for r in rs], [r["cell"] for r in rs]
    total = slope([r["coef"] for r in rs], y, cl, fe)
    assoc = slope([r["mon"][t] for r in rs], y, cl, fe)
    direct, via_readout = ols(y, [[r["coef"], r["mon"][t]] for r in rs], cl, fe)
    share = (1 - direct["b"] / total["b"]) if total["b"] and math.isfinite(total["b"]) and total["p"] < 0.05 \
        else float("nan")
    return {"total": total, "assoc": assoc, "direct": direct, "via_readout": via_readout, "share": share}


def obs_compare(rows, t, value_slope):
    """Text vs image on the games played in both modes: (text slope, image slope, difference)."""
    groups = defaultdict(set)
    for r in rows:
        groups[r["group"]].add(r["obs"])
    both = {g for g, modes in groups.items() if {"text", "image"} <= modes}
    sub = [r for r in rows if r["group"] in both]
    res = {mode: value_slope([r for r in sub if r["obs"] == mode], t) for mode in ("text", "image")}
    a, b = res["text"], res["image"]
    if math.isfinite(a.get("b", float("nan"))) and math.isfinite(b.get("b", float("nan"))):
        se = math.sqrt(a["se"] ** 2 + b["se"] ** 2) if math.isfinite(a["se"]) and math.isfinite(b["se"]) else float("nan")
        d = b["b"] - a["b"]
        res["diff"] = {"n": a["n"] + b["n"], "clusters": a.get("clusters", 0) + b.get("clusters", 0), "b": d, "se": se,
                       "p": math.erfc(abs(d / se) / math.sqrt(2)) if se and math.isfinite(se) and se > 0 else float("nan")}
    else:
        res["diff"] = {"n": 0, "b": float("nan"), "se": float("nan"), "p": float("nan")}
    return res, sorted(both)


def paired(rows, base, select):
    """Rows chosen by `select`, each with its paired reward difference (same seat, same seed, unsteered)."""
    out = []
    for r in rows:
        if r["condition"] == "none" or r["reward"] is None or not select(r):
            continue
        b = base.get((r["group"], r["obs"], r["episode"], r["agent"]))
        if b is not None:
            out.append({**r, "diff": r["reward"] - b})
    return out


def reward_scale(rows):
    """Per game cell: the SD of the unsteered agents' rewards (1 when it is 0 or unknown)."""
    vals = defaultdict(list)
    for r in rows:
        if r["condition"] == "none" and r["reward"] is not None:
            vals[r["cell"]].append(r["reward"])
    return {c: (float(np.std(v)) or 1.0) if len(v) > 1 else 1.0 for c, v in vals.items()}


def effect_table(prs, traits, scale):
    """{trait: {"pooled": slope (rewards in unsteered-SD units, game FE), cell: slope (raw rewards)}}."""
    table = {}
    for t in traits:
        tr = [r for r in prs if r["trait"] == t and r["coef"] is not None]
        res = {"pooled": slope([r["coef"] for r in tr], [r["diff"] / scale.get(r["cell"], 1.0) for r in tr],
                               [r["cluster"] for r in tr], [r["cell"] for r in tr])}
        for cell in sorted({r["cell"] for r in tr}):
            cr = [r for r in tr if r["cell"] == cell]
            res[cell] = slope([r["coef"] for r in cr], [r["diff"] for r in cr], [r["cluster"] for r in cr])
        table[t] = res
    return table


def fingerprint(rs):
    """Mean readout of every monitored trait (SD units vs unsteered agents of the same game) over the agents `rs`:
    which traits an intervention moves, and how much."""
    vals = defaultdict(list)
    for r in rs:
        for u, v in r["mon"].items():
            vals[u].append(v)
    return {u: float(np.mean(v)) for u, v in vals.items()}


def cosine(a, b):
    keys = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    n = float(np.linalg.norm(x) * np.linalg.norm(y)) if keys else 0.0
    return float(x @ y / n) if n > 0 else float("nan")


def pearson(x, y):
    """(r, low, high, n): correlation with a 95% CI (Fisher z)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), float("nan"), len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    z, h = math.atanh(max(-0.9999, min(0.9999, r))), 1.96 / math.sqrt(len(x) - 3)
    return r, math.tanh(z - h), math.tanh(z + h), len(x)


def equivalent_coef(curve, target, sign):
    """The vector coefficient on the `sign` side whose mean readout equals `target` (linear interpolation of the
    dose-response curve {coef: mean}, 0 = unsteered). Returns (coef, note): note '>' when even the largest
    coefficient stays below the target, '<' when the target is not past the unsteered level."""
    pts = sorted(((x, m) for x, m in curve.items() if x * sign >= 0 and math.isfinite(m)), key=lambda p: abs(p[0]))
    if len(pts) < 2 or not math.isfinite(target):
        return float("nan"), ""
    for (x0, m0), (x1, m1) in zip(pts, pts[1:]):
        if (m0 - target) * (m1 - target) <= 0 and m1 != m0:
            return x0 + (target - m0) / (m1 - m0) * (x1 - x0), ""
    if sign * (target - pts[0][1]) <= 0:
        return 0.0, "<"
    return pts[-1][0], ">"


def channels(rows, traits, base, scale):
    """Persona prompt vs persona vector, per trait and sign (+ = the trait's instruction / c > 0): strength,
    direction over all monitored traits, behaviour, visibility and cost. Vector numbers at the largest
    coefficient of that sign unless said otherwise; only game cells that have both channels."""
    wel = welfare([r for r in rows if r["condition"] in ("all", "prompt_all")], [r for r in rows if r["condition"] == "none"])
    out, agree = {}, defaultdict(list)  # agree: [(prompt, vector, vector at the next-lower |c|, label)] per measure
    for t in traits:
        vec = [r for r in rows if r["steered"] and r["trait"] == t and r["via"] == "vector" and not r["random"]
               and r["team_cond"] is None]
        pro = [r for r in rows if r["steered"] and r["trait"] == t and r["via"] == "prompt" and r["team_cond"] is None]
        cells = {r["cell"] for r in vec} & {r["cell"] for r in pro}
        vec, pro = [r for r in vec if r["cell"] in cells], [r for r in pro if r["cell"] in cells]
        rnd = [r for r in rows if r["steered"] and r["trait"] == t and r["random"] and r["cell"] in cells]
        none = [r for r in rows if r["condition"] == "none" and r["cell"] in cells]
        if not cells:
            continue
        curve = {0.0: _mean_mon(none, t)} | {c: _mean_mon([r for r in vec if r["coef"] == c], t)
                                               for c in {r["coef"] for r in vec}}
        coefs = sorted(c for c in curve if c)
        res = {"cells": sorted(cells)}
        for s in (1, -1):
            side = [c for c in coefs if c * s > 0]
            if not side:
                continue
            cmax, cnext = max(side, key=abs), sorted(side, key=abs)[-2] if len(side) > 1 else None
            p_s = [r for r in pro if (r["coef"] or 0) * s > 0]
            v_max = [r for r in vec if r["coef"] == cmax]
            shift = mean_ci([r["mon"][t] for r in p_s if t in r["mon"]], [r["cluster"] for r in p_s if t in r["mon"]])
            fp_p, fp_v, fp_r = fingerprint(p_s), fingerprint(v_max), fingerprint([r for r in rnd if r["coef"] == cmax])
            rank = lambda fp: 1 + sorted(fp, key=lambda u: -s * fp[u]).index(t) if t in fp else None
            rew = paired(rows, base, lambda r: r["steered"] and r["trait"] == t and r["cell"] in cells)
            rew_mean = lambda cond, c, team=None: mean_ci(
                *zip(*[(r["diff"] / scale.get(r["cell"], 1.0), r["cluster"]) for r in rew
                       if r["condition"] == cond and r["coef"] == c and (team is None or r["team"] == team)]
                     or [(float("nan"), None)]))
            wel_mean = lambda cond, c: mean_ci(
                *zip(*[(r["diff"], r["cluster"]) for r in wel
                       if r["condition"] == cond and r["trait"] == t and r["coef"] == c and r["cell"] in cells]
                     or [(float("nan"), None)]))
            co = lambda cond, sel: [r for r in rows if r["condition"] == cond and not r["steered"] and r["trait"] == t
                                    and sel(r["coef"]) and r["cell"] in cells and t in r["mon"]]
            contagion = {k: mean_ci([r["mon"][t] for r in rs], [r["cluster"] for r in rs]) for k, rs in
                         (("prompt", co("prompt_one", lambda c: (c or 0) * s > 0)), ("vector", co("one", lambda c: c == cmax)))}
            det = {"prompt": detectability(rows, t, ("prompt",)).get(float(s)),
                   "vector": detectability(rows, t, ("vector",)).get(cmax),
                   "both": detectability(rows, t, ("both",)).get(cmax)}
            cost = lambda rs: (mean_ci([r["chars"] for r in rs if r["chars"] is not None],
                                       [r["cluster"] for r in rs if r["chars"] is not None])[0],
                               100 * sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs)))
            res[s] = {"shift": shift, "eq": equivalent_coef(curve, shift[0], s), "cmax": cmax, "vmax": curve[cmax],
                      "cos_vec": cosine(fp_p, fp_v), "cos_rnd": cosine(fp_p, fp_r),
                      "cos_vec_rnd": cosine(fp_v, fp_r), "top_p": _top(fp_p, s), "top_v": _top(fp_v, s),
                      "rank_p": rank(fp_p), "rank_v": rank(fp_v),
                      "reward": {"prompt": rew_mean("prompt_one", float(s)), "vector": rew_mean("one", cmax),
                                 "vector_next": rew_mean("one", cnext)},
                      "evil": {"prompt": rew_mean("evil_team_prompt", float(s), "evil"),
                               "vector": rew_mean("evil_team", cmax, "evil")},
                      "welfare": {"prompt": wel_mean("prompt_all", float(s)), "vector": wel_mean("all", cmax)},
                      "contagion": contagion, "det": det,
                      "cost": {"prompt": cost(p_s), "vector": cost(v_max), "none": cost(none)}}
            for cell in sorted(cells):
                in_cell = lambda rs: [r for r in rs if r["cell"] == cell]
                label = f"{t} {'+' if s > 0 else '−'} {cell_name(cell)}"
                agree["readout"].append((_mean_mon(in_cell(p_s), t), _mean_mon(in_cell(v_max), t),
                                         _mean_mon(in_cell([r for r in vec if r["coef"] == cnext]), t), label))
                cr = [r for r in rew if r["cell"] == cell]
                m = lambda cond, c: float(np.mean([r["diff"] / scale.get(cell, 1.0) for r in cr
                                                   if r["condition"] == cond and r["coef"] == c] or [float("nan")]))
                agree["reward"].append((m("prompt_one", float(s)), m("one", cmax), m("one", cnext), label))
        out[t] = res
    corr = {k: {"prompt_vector": pearson([a for a, *_ in v], [b for _, b, *_ in v]),
                "vector_vector": pearson([b for _, b, *_ in v], [c for *_, c, _ in v])} for k, v in agree.items()}
    return out, agree, corr


def _mean_mon(rs, t):
    v = [r["mon"][t] for r in rs if t in r["mon"]]
    return float(np.mean(v)) if v else float("nan")


def _top(fp, sign, k=3):
    """The k monitored traits an intervention moves most in the direction `sign`."""
    return [(u, fp[u]) for u in sorted(fp, key=lambda u: -sign * fp[u])[:k]]


# ---------------------------------------------------------------- HTML

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e6; --pos:#c0392b; --neg:#2c6fbb; --card:#f6f6f7;
  --rnd:#8a8a90; --warn:#fff4d6; --ok:#e9f6ec; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececee; --muted:#9a9aa1; --line:#2c2c30;
  --card:#1e1e21; --warn:#3a3220; --ok:#1d3326; } }
body { background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, sans-serif; margin:0 auto; max-width:1180px;
  padding:16px; }
h1 { font-size:26px; margin-bottom:4px; } h2 { margin-top:44px; border-bottom:2px solid var(--line); padding-bottom:4px; }
h3 { font-size:15px; margin:18px 0 4px; }
p, li { max-width:920px; } .muted { color:var(--muted); }
.box { display:grid; grid-template-columns:150px 1fr; gap:4px 14px; background:var(--card); border-radius:8px;
  padding:10px 14px; margin:8px 0 12px; max-width:1000px; }
.box b { color:var(--muted); font-weight:600; }
.result { background:var(--ok); border-radius:8px; padding:10px 14px; margin:10px 0; max-width:1000px; }
.result ul { margin:4px 0; padding-left:20px; }
.warn { background:var(--warn); border-radius:8px; padding:10px 14px; margin:10px 0; max-width:1000px; }
table { border-collapse:collapse; margin:6px 0 14px; font-variant-numeric:tabular-nums; font-size:13.5px; }
th, td { border:1px solid var(--line); padding:3px 7px; text-align:right; } th { background:var(--card); }
td.l, th.l { text-align:left; }
.wrap { overflow-x:auto; }
.charts { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:12px; }
.chart { background:var(--card); border-radius:8px; padding:8px; }
.chart h3 { margin:0 0 2px; }
svg text { fill:var(--muted); font-size:10px; }
.sig { font-weight:700; }
.legend span { display:inline-block; margin-right:14px; font-size:12px; }
.legend i { display:inline-block; width:18px; height:3px; vertical-align:middle; margin-right:5px; }
"""


def fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def stars(q):
    return "" if not (q == q) else "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def sig(s, q=None):
    p = q if q is not None else (s or {}).get("p", float("nan"))
    return bool(s) and math.isfinite(s.get("b", float("nan"))) and p == p and p < 0.05


def heat(b, se):
    """Cell background by the slope's t value (red = up, blue = down)."""
    if not (se and math.isfinite(se) and se > 0 and math.isfinite(b)):
        return ""
    t = max(-1.0, min(1.0, (b / se) / 4))
    colour = "192,57,43" if t > 0 else "44,111,187"
    return f' style="background:rgba({colour},{abs(t) * 0.55:.2f})"'


def slope_cell(s, q=None):
    if not s or not math.isfinite(s.get("b", float("nan"))):
        return '<td class=muted title="not enough data">–</td>'
    tip = f'n={s["n"]}, episodes={s.get("clusters", "?")}, SE={fmt(s["se"])}, p={fmt(s["p"], 4)}' + \
          (f", q={fmt(q, 4)}" if q is not None else "")
    return f'<td{heat(s["b"], s["se"])} title="{tip}">{fmt(s["b"], 2)}<span class="sig">{stars(q if q is not None else s["p"])}</span></td>'


def line_chart(series, title, ylabel, xlabel="steering coefficient", width=300, height=200):
    """series: {name: [(x, mean, half_ci)]} -> SVG with axes, labels and a zero line."""
    pts = [p for s in series.values() for p in s if math.isfinite(p[1])]
    if not pts:
        return "<p class=muted>no data yet</p>"
    xs = [p[0] for p in pts]
    lo = min([p[1] - (p[2] if math.isfinite(p[2]) else 0) for p in pts] + [0])
    hi = max([p[1] + (p[2] if math.isfinite(p[2]) else 0) for p in pts] + [0])
    lo, hi = (lo - 1, hi + 1) if hi == lo else (lo, hi)
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    L, R, T, B = 44, 8, 8, 34
    sx = lambda x: L + (x - x0) / (x1 - x0) * (width - L - R)
    sy = lambda y: T + (hi - y) / (hi - lo) * (height - T - B)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{height - B}" stroke="currentColor" stroke-opacity=".3"/>',
             f'<line x1="{L}" y1="{sy(0):.1f}" x2="{width - R}" y2="{sy(0):.1f}" stroke="currentColor" '
             f'stroke-opacity=".3" stroke-dasharray="2 2"/>']
    for v in (lo, 0, hi):
        parts.append(f'<text x="{L - 4}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:.2g}</text>')
    for v in sorted(set(xs)):
        parts.append(f'<text x="{sx(v):.1f}" y="{height - B + 12}" text-anchor="middle">{v:g}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text transform="translate(10,{(T + height - B) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 f'{html.escape(ylabel)}</text>')
    colours = {"vector": "var(--pos)", "random": "var(--rnd)", "prompt": "var(--neg)"}
    for name, s in series.items():
        s = sorted(p for p in s if math.isfinite(p[1]))
        c = colours.get(name, "currentColor")
        dash = ' stroke-dasharray="4 3"' if name == "random" else ""
        if len(s) > 1 and name != "prompt":
            parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.8"{dash} points="'
                         + " ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m, _ in s) + '"/>')
        for x, m, h in s:
            if math.isfinite(h):
                parts.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(m - h):.1f}" y2="{sy(m + h):.1f}" '
                             f'stroke="{c}" stroke-opacity=".6"/>')
            shape = (f'<rect x="{sx(x) - 3.5:.1f}" y="{sy(m) - 3.5:.1f}" width="7" height="7" fill="{c}">'
                     if name == "prompt" else f'<circle cx="{sx(x):.1f}" cy="{sy(m):.1f}" r="2.8" fill="{c}">')
            parts.append(shape + f'<title>{name}, c={x:g}: {m:.2f} ± {h:.2f}</title>'
                         + ("</rect>" if name == "prompt" else "</circle>"))
    parts.append("</svg>")
    return "".join(parts)


def scatter_chart(points, xlabel, ylabel, width=320, height=280):
    """points: [(x, y, label)] -> SVG scatter with zero lines and the diagonal y = x (same effect both ways)."""
    pts = [(x, y, lab) for x, y, lab in points if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return "<p class=muted>no data yet</p>"
    lo = min(min(x, y) for x, y, _ in pts + [(0, 0, "")])
    hi = max(max(x, y) for x, y, _ in pts + [(0, 0, "")])
    pad = 0.05 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 44, 8, 8, 34
    sx = lambda v: L + (v - lo) / (hi - lo) * (width - L - R)
    sy = lambda v: T + (hi - v) / (hi - lo) * (height - T - B)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(ylabel)}">',
             f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="currentColor" '
             f'stroke-opacity=".35" stroke-dasharray="4 3"/>',
             f'<line x1="{L}" y1="{sy(0):.1f}" x2="{width - R}" y2="{sy(0):.1f}" stroke="currentColor" stroke-opacity=".2"/>',
             f'<line x1="{sx(0):.1f}" y1="{T}" x2="{sx(0):.1f}" y2="{height - B}" stroke="currentColor" stroke-opacity=".2"/>']
    for v in (lo, 0, hi):
        parts.append(f'<text x="{L - 4}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:.2g}</text>')
        parts.append(f'<text x="{sx(v):.1f}" y="{height - B + 12}" text-anchor="middle">{v:.2g}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text transform="translate(10,{(T + height - B) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 f'{html.escape(ylabel)}</text>')
    for x, y, lab in pts:
        c = "var(--pos)" if "+" in lab else "var(--neg)"
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{c}" fill-opacity=".75">'
                     f'<title>{html.escape(lab)}: x={x:.2f}, y={y:.2f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


LEGEND = ('<div class=legend><span><i style="background:var(--pos)"></i>persona vector</span>'
          '<span><i style="background:var(--rnd)"></i>random direction, same norm (control)</span>'
          '<span><i style="background:var(--neg);height:7px;width:7px"></i>persona prompt (+ / − instruction)</span>'
          '<span>bars: 95% CI</span></div>')


def cell_name(cell):
    group, obs = cell
    return group if obs == "text" else f"{group} ({obs})"


def section(num, title, question, measured, read, findings, body, warn=None):
    """One analysis: what it asks, what exactly is measured, how to read it, what it shows, the numbers."""
    items = "".join(f"<li>{f}</li>" for f in findings) or "<li>not enough data yet</li>"
    return (f"<h2 id=s{num}>{num}. {html.escape(title)}</h2>"
            f"<div class=box><b>Question</b><span>{question}</span><b>What is measured</b><span>{measured}</span>"
            f"<b>How to read it</b><span>{read}</span></div>"
            + (f"<div class=warn>{warn}</div>" if warn else "")
            + f"<div class=result><b>Result</b><ul>{items}</ul></div>{body}")


def mean_cell(v):
    """A (mean, half CI, n) as a table cell: 'm ± h', or 'm (n=1)' when one episode gives no CI."""
    if not v or not math.isfinite(v[0]):
        return "<td class=muted>–</td>"
    return f"<td>{fmt(v[0], 2)} ± {fmt(v[1], 2)}</td>" if math.isfinite(v[1]) else         f"<td title='a single episode: no CI'>{fmt(v[0], 2)} <span class=muted>(n=1)</span></td>"


def ranked(d, k=3, reverse=True):
    """'a (1.2), b (0.8)' for the k largest (or smallest) finite values of {name: value}."""
    items = sorted(((n, v) for n, v in d.items() if v == v), key=lambda kv: kv[1], reverse=reverse)[:k]
    return ", ".join(f"{n} ({v:+.2f})" for n, v in items)


def sig_cells(found, q, fmt_key=str):
    """Significant cells of a matrix as 'key: +0.12' strings, strongest first."""
    hits = [(k, s["b"]) for k, s in found.items() if sig(s, q.get(k))]
    hits.sort(key=lambda kv: -abs(kv[1]))
    return [f"{fmt_key(k)}: {b:+.2f}" for k, b in hits]


def matrix(title, corner, row_keys, col_keys, get):
    """(html, {key: slope}, {key: q}) for a rows x columns table of slopes, BH-corrected across its cells."""
    keys = [(a, b) for a in row_keys for b in col_keys]
    found = {k: get(*k) for k in keys}
    q = dict(zip(keys, bh([found[k]["p"] if found[k] else float("nan") for k in keys])))
    out = [f"<h3>{html.escape(title)}</h3>" if title else "",
           f"<div class=wrap><table><tr><th class=l>{html.escape(corner)}</th>"
           + "".join(f"<th>{html.escape(cell_name(c) if isinstance(c, tuple) else str(c))}</th>" for c in col_keys)
           + "</tr>"]
    for a in row_keys:
        out.append(f"<tr><td class=l>{html.escape(cell_name(a) if isinstance(a, tuple) else str(a))}</td>"
                   + "".join(slope_cell(found[(a, b)], q[(a, b)]) for b in col_keys) + "</tr>")
    out.append("</table></div>")
    return "".join(out), found, q


def _diff_slope(rs, scale):
    """Slope of a paired difference over the coefficient (scaled by the game's unsteered SD if `scale`)."""
    rs = [r for r in rs if r["coef"] is not None]
    return slope([r["coef"] for r in rs], [r["diff"] / (scale.get(r["cell"], 1.0) if scale else 1.0) for r in rs],
                 [r["cluster"] for r in rs], [r["cell"] for r in rs])


def setup_section(rows, sweep, traits, cells):
    """What was run: model, steering, conditions, games, monitor, statistics."""
    coefs = sorted({r["coef"] for r in rows if r["coef"] is not None and r["via"] != "prompt"})
    conds = sorted({r["condition"] for r in rows})
    eps = len({r["cluster"] for r in rows})
    base_eps = defaultdict(set)
    for r in rows:
        if r["condition"] == "none":
            base_eps[r["cell"]].add(r["episode"])
    layers = defaultdict(set)
    for r in rows:
        if r.get("steer_layer") is not None and r["trait"]:
            layers[r["trait"]].add(int(r["steer_layer"]))
    mon_layer = sorted({r["monitor_layer"] for r in rows if r.get("monitor_layer") is not None})
    sw = sweep or {}
    cond_help = {
        "none": "nobody steered (the baseline every comparison is made against)",
        "one": "one randomly drawn agent gets the persona vector", "half": "half of the agents (drawn at random)",
        "all": "every agent", "all_random": "every agent gets a random direction of the same norm (control)",
        "random_one": "one agent gets a random direction of the same norm (control)",
        "prompt_one": "one agent gets the trait's persona instruction in its system prompt (no vector)",
        "prompt_all": "every agent gets the persona instruction", "both_one": "one agent gets the instruction AND the vector",
        "evil_team": "every member of the deceiving side (hidden-role games)", "evil_one": "one member of the deceiving side",
        "good_one": "one member of the honest side", "evil_team_prompt": "the deceiving side gets the persona instruction",
        "vector_one": "one agent gets the persona vector", "vector_all": "every agent gets the persona vector",
        "evil_team_vector": "every member of the deceiving side gets the vector"}
    gen = ", ".join(f"{k} {sw[k]}" for k in ("max_tokens", "max_turns", "max_parse_retries") if k in sw)
    out = ["<h2 id=setup>Setup: what was run and what is measured</h2><div class=box>",
           f"<b>Model</b><span>{html.escape(str(sw.get('model', '(see the sweep file)')))}; every agent is the same "
           f"model, generating with PersVecFast's sampling settings (Qwen3.5: temperature 0.7, top-p 0.8, top-k 20, "
           f"presence penalty 1.5, thinking off){'; ' + html.escape(gen) if gen else ''}.</span>",
           "<b>Steering</b><span>vLLM-Hook's <code>add_vector</code>: at one decoder block, every token's hidden state "
           "becomes <i>h + c·v</i>, with <i>v</i> the raw PersVecFast persona vector of the trait (mean difference "
           "between trait-expressing and trait-suppressing answers) and <i>c</i> the steering coefficient. Steered "
           "and unsteered agents share one batch; each request carries its own vector. "
           + (f"Layers: {', '.join(f'{t} L{min(v)}' for t, v in sorted(layers.items()))} (the trait's best layer in "
              f"PersVecFast's steering sweep)." if layers else "Layer: the trait's best layer in PersVecFast's steering "
              "sweep.") + "</span>",
           f"<b>Traits</b><span>{', '.join(traits)}</span>",
           f"<b>Coefficients</b><span>{', '.join(f'{c:g}' for c in coefs)} (negative = push away from the trait). "
           f"Persona prompts use only the sign: + = the trait's instruction, − = its opposite.</span>",
           "<b>Conditions</b><span><table>" + "".join(
               f"<tr><td class=l><code>{c}</code></td><td class=l>{cond_help.get(c, '')}</td></tr>" for c in conds)
           + "</table></span>",
           f"<b>Games</b><span>{len(cells)} (game × observation mode), {eps} episodes in this report. Image mode: the "
           f"game state is shown as a picture instead of text. Episode seeds depend only on the game and the "
           f"episode number, so every condition plays the same deals, roles and seats.</span>",
           "<b>Monitor (readout)</b><span>A second, <i>unsteered</i> copy of the model reads each agent's prompt and "
           "answer. Its hidden state after the monitor layer"
           + (f" (block {', '.join(str(m) for m in mon_layer)})" if mon_layer else "")
           + " is averaged over the answer's tokens and projected onto each trait's unit vector: how strongly the "
           "model represents that trait in what the agent wrote. Because the reader is unsteered, it measures the "
           "text, not the injected vector. Reported in SD units of the unsteered agents of the same game "
           "(0 = typical unsteered agent, +1 = one standard deviation more of the trait).</span>",
           "<b>Outcome</b><span>Each game's own reward for the agent (e.g. win/loss of its side in Avalon, Werewolf, "
           "Mafia; points or goods won in auctions, harvest, TextArena games). Compared <i>paired</i>: the same seat in "
           "the unsteered episode with the same seed (same deal, same role).</span>",
           "<b>Statistics</b><span>Effects are slopes per unit coefficient (a straight line through the condition's "
           "points and the unsteered agents at 0), with games as fixed effects when pooled. Standard errors are "
           "clustered by episode. <b>*</b> q&lt;.05, <b>**</b> q&lt;.01, <b>***</b> q&lt;.001 after Benjamini–Hochberg "
           "correction over all cells of a table. Cell colour: red = increase, blue = decrease, stronger = more "
           "certain. Hover a cell for n, SE, p, q; '–' = not enough data.</span></div>"]
    few = {c: len(v) for c, v in base_eps.items() if len(v) < 10}
    if few:
        out.append("<div class=warn><b>Few unsteered episodes so far</b> in "
                   + ", ".join(f"{html.escape(cell_name(c))} ({n})" for c, n in sorted(few.items()))
                   + ". Every SD unit and every paired comparison rests on these, so effect <i>sizes</i> are noisy "
                     "and may be inflated; trust directions and rankings more than magnitudes.</div>")
    return "".join(out)


def _turns(path):
    try:
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    except OSError:
        return []


def games_section(rows, traits, cells, man, eff, base, scale):
    """Every game on its own: per trait, the language (vector slope, prompt shift) and the outcomes (steered
    agent, prompted agent, the group, the deceiving side), in the game's own reward units."""
    wel = welfare([r for r in rows if r["condition"] == "all" and not r["random"]],
                  [r for r in rows if r["condition"] == "none"])
    evil = paired(rows, base, lambda r: r["team_cond"] == "evil" and r["team"] == "evil" and r["steered"])
    prompted = paired(rows, base, lambda r: r["condition"] == "prompt_one" and r["steered"])
    # per setting: the steered agents' and their unsteered co-players' paired reward over the coefficient
    own_all = paired(rows, base, lambda r: r["steered"] and r["trait"] is not None)
    co_all = paired(rows, base, lambda r: not r["steered"] and r["trait"] is not None)
    out = ["<h2 id=games>Per game</h2><div class=box><b>What you see</b><span>Each game separately, one row per "
           "trait. <b>Readout</b>: the vector's slope on the steered agents' trait readout (SD per unit c; section 2) "
           "and the mean readout with the persona prompt + / − (SD). <b>Outcomes</b>, in the game's own reward units, "
           "paired with the same seat in the unsteered episode with the same seed: the steered agent's reward per "
           "unit c (vector, section 9); the one prompted agent's mean reward change (prompt + / −); the group's "
           "total reward per unit c when every agent gets the vector (welfare); in hidden-role games the deceiving "
           "side's reward per unit c when all its members get the vector. Below each game, <b>by setting</b>: the "
           "same reward slope for every setting (who is steered: one, half, all, one side, the random control, the "
           "prompt), for the steered agents and for the unsteered co-players of the same episodes.</span><b>How to read it</b><span>Slopes "
           "carry stars after Benjamini–Hochberg correction over the game's table; prompt columns are means ± 95% CI. "
           "The unsteered line gives the scale: e.g. a win/loss game with reward 0/1 has mean reward = win rate, so "
           "a slope of 0.1 = +10 percentage points of wins per unit c.</span></div>"]
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell]
        unst = [r for r in cr if r["condition"] == "none"]
        if not any(r["trait"] for r in cr):
            continue
        if not unst:
            out.append(f"<h3>{html.escape(cell_name(cell))}</h3><p class=muted>{len({r['cluster'] for r in cr})} "
                       "episodes, none unsteered yet: nothing to compare against.</p>")
            continue
        has_evil = any(r["team"] == "evil" for r in cr)
        ev_slope = {t: _diff_slope([r for r in evil if r["cell"] == cell and r["trait"] == t], None) for t in traits}
        wel_slope = {t: _diff_slope([r for r in wel if r["cell"] == cell and r["trait"] == t], None) for t in traits}
        cols = [("readout", lambda t: man[t]["per_game"].get(cell)), ("reward", lambda t: eff[t].get(cell)),
                ("welfare", lambda t: wel_slope[t])] + ([("evil", lambda t: ev_slope[t])] if has_evil else [])
        found = {(name, t): get(t) for name, get in cols for t in traits}
        q = dict(zip(found, bh([s["p"] if s else float("nan") for s in found.values()])))
        pm = lambda rs: mean_ci([v for v, _ in rs], [c for _, c in rs])

        tab = ["<div class=wrap><table><tr><th class=l>trait</th><th>readout: vector slope</th><th>readout: prompt +</th>"
               "<th>readout: prompt −</th><th>steered agent's reward: vector slope</th><th>prompted agent's reward: +</th>"
               "<th>prompted agent's reward: −</th><th>welfare: vector slope (all steered)</th>"
               + ("<th>evil side's reward: vector slope</th>" if has_evil else "") + "</tr>"]
        hits = []
        for t in traits:
            pr = {sgn: pm([(r["mon"][t], r["cluster"]) for r in cr if r["via"] == "prompt" and r["steered"] and r["trait"] == t
                           and t in r["mon"] and r["team_cond"] is None and (r["coef"] or 0) * sgn > 0]) for sgn in (1, -1)}
            rw = {sgn: pm([(r["diff"], r["cluster"]) for r in prompted if r["cell"] == cell and r["trait"] == t
                           and (r["coef"] or 0) * sgn > 0]) for sgn in (1, -1)}
            tab.append(f"<tr><td class=l>{t}</td>{slope_cell(found[('readout', t)], q[('readout', t)])}{mean_cell(pr[1])}{mean_cell(pr[-1])}"
                       f"{slope_cell(found[('reward', t)], q[('reward', t)])}{mean_cell(rw[1])}{mean_cell(rw[-1])}"
                       f"{slope_cell(found[('welfare', t)], q[('welfare', t)])}"
                       + (slope_cell(found[("evil", t)], q[("evil", t)]) if has_evil else "") + "</tr>")
            for name, label in (("reward", "steered agent"), ("welfare", "group"), ("evil", "evil side")):
                s = found.get((name, t))
                if s and sig(s, q[(name, t)]):
                    hits.append(f"{t} → {label} {s['b']:+.2f}/unit c")
        tab.append("</table></div>")
        rew0 = [r["reward"] for r in unst if r["reward"] is not None]
        evil0 = [r["reward"] for r in unst if r["reward"] is not None and r["team"] == "evil"]
        out.append(f"<h3>{html.escape(cell_name(cell))}</h3><p class=muted>{len({r['cluster'] for r in cr})} episodes "
                   f"({len({r['cluster'] for r in unst})} unsteered). Unsteered: mean reward per agent "
                   f"{fmt(float(np.mean(rew0)) if rew0 else float('nan'), 2)} (SD {fmt(scale.get(cell, float('nan')), 2)})"
                   + (f", evil side {fmt(float(np.mean(evil0)), 2)}" if evil0 else "") + ". "
                   + ("<b>The deceiving side wins every unsteered game</b>, so steering can only make it lose: a "
                      "positive evil-side slope means it loses at negative c (pushed away from the trait), not that "
                      "it wins more at positive c. " if evil0 and min(evil0) >= 1 else "")
                   + ("Significant outcome effects: " + "; ".join(hits) + "." if hits else "No significant outcome effect yet.")
                   + "</p>" + "".join(tab) + settings_tables(cell, traits, own_all, co_all))
    return "".join(out)


SETTING_ORDER = ("one", "half", "all", "both_one", "good_one", "evil_one", "evil_team", "all_random",
                 "prompt_one", "prompt_all", "evil_team_prompt")


def settings_tables(cell, traits, own_all, co_all) -> str:
    """One game: reward slope per unit c (raw units, paired with the same seat unsteered) per setting × trait,
    for the steered agents and for the unsteered co-players. Prompt settings have only a sign (c = ±1)."""
    own = [r for r in own_all if r["cell"] == cell]
    co = [r for r in co_all if r["cell"] == cell]
    conds = sorted({r["condition"] for r in own},
                   key=lambda c: (SETTING_ORDER.index(c) if c in SETTING_ORDER else len(SETTING_ORDER), c))
    if not conds:
        return ""
    t_own, f_own, q_own = matrix("By setting: steered agents' reward per unit c", "setting", conds, traits,
                                 lambda c, t: _diff_slope([r for r in own if r["condition"] == c and r["trait"] == t], None))
    co_conds = [c for c in conds if any(r["condition"] == c for r in co)]
    t_co, f_co, q_co = matrix("By setting: unsteered co-players' reward per unit c", "setting", co_conds, traits,
                              lambda c, t: _diff_slope([r for r in co if r["condition"] == c and r["trait"] == t], None))
    hits = sig_cells(f_own, q_own, lambda k: f"{k[1]} ({k[0]}): steered") +         sig_cells(f_co, q_co, lambda k: f"{k[1]} ({k[0]}): co-players")
    how = ("<p class=muted><b>How to read:</b> a cell is the change in this game's own reward per +1 of steering "
           "coefficient, for the agents in that setting, against the same seat in the unsteered game with the same "
           "seed. E.g. +0.10 in a win/loss game = +10 points of win rate per unit c, so +20 points at c = 2 and −20 at "
           "c = −2. Top table: the steered agents; bottom: the agents of the same episodes that were not steered "
           "(what the steered ones do to them). Prompt settings have no strength, only + vs − instruction (c = ±1). "
           "Stars: Benjamini–Hochberg over the table; '–' = too little data.</p>")
    note = how + (("<p class=muted>By setting, significant: " + "; ".join(hits[:10])
                   + (f"; {len(hits) - 10} more" if len(hits) > 10 else "") + ".</p>") if hits else
                  "<p class=muted>By setting: no significant reward slope.</p>")
    return "<details open><summary>By setting (one, half, all, …)</summary>" + note + t_own + t_co + "</details>"


def examples_section(logs_dir, traits, coefs, per_trait=2, chars=700):
    """What steering looks like: the same seat's first turn of the same game (same seed, so the same prompt)
    unsteered, with the persona vector at the strongest - and + coefficient, the random direction and the
    persona prompt (+ / -). Conditions where every agent is steered (all, all_random, prompt_all), so the
    first-turn agent is steered in each."""
    logs = Path(logs_dir)
    if not logs.is_dir() or not coefs:
        return ""
    lo, hi = min(coefs), max(coefs)
    out = ["<h2 id=examples>What steering looks like: matched examples</h2><div class=box><b>What you see</b><span>"
           "For each trait, games where the same seat's <i>first</i> answer can be compared across conditions: same "
           "game, same seed, so the agent received exactly the same prompt; the only difference is the steering. "
           f"Columns: unsteered; persona vector at c = {lo:g} and {hi:g}; a random direction of the same norm at "
           f"c = {hi:g} (control); the persona prompt (+ / −). The number under each answer is that turn's readout "
           "of the trait (raw projection, higher = more of the trait; compare within a row).</span>"
           "<b>Why it matters</b><span>The tables below are averages over thousands of turns; these show what a "
           "slope of the readout actually corresponds to in text, including where strong steering stops being "
           "'more of the trait' and becomes caricature or broken output.</span></div>"]
    for t in traits:
        shown = 0
        for group_dir in sorted(p for p in logs.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if shown >= per_trait:
                break
            g = group_dir.name
            runs = {"unsteered": f"{g}__none", f"vector c={lo:g}": f"{g}__all__{t}_x{lo:g}",
                    f"vector c={hi:g}": f"{g}__all__{t}_x{hi:g}", f"random c={hi:g}": f"{g}__all_random__{t}_x{hi:g}",
                    "prompt +": f"{g}__prompt_all__{t}_x1", "prompt −": f"{g}__prompt_all__{t}_x-1"}
            eps = None
            for run in runs.values():
                have = {int(p.name.split("_")[1].split(".")[0]) for p in (group_dir / run).glob("episode_*.summary.json")}
                eps = have if eps is None else eps & have
            if not eps:
                continue
            ep = min(eps)
            first = {k: (_turns(group_dir / run / f"episode_{ep}.jsonl") or [None])[0] for k, run in runs.items()}
            base = first["unsteered"]
            if not base or len(base.get("completion") or "") < 120:  # one-word moves say little about style
                continue
            cells = []
            for k, rec in first.items():
                if not rec:
                    cells.append(f"<td class=l><b>{k}</b><br><span class=muted>missing</span></td>")
                    continue
                mon = rec.get("monitor") or {}
                layer = sorted(mon, key=int)[len(mon) // 2] if mon else None
                score = mon.get(layer, {}).get(t) if layer else None
                text = (rec.get("completion") or "").strip()
                text = html.escape(text[:chars] + (" …" if len(text) > chars else "")).replace("\n", "<br>")
                cells.append(f"<td class=l style='vertical-align:top;min-width:220px'><b>{k}</b><br>{text}<br>"
                             f"<span class=muted>{t} readout: {fmt(score, 2)}</span></td>")
            out.append(f"<h3>{t} — {html.escape(g)}, episode {ep}, {html.escape(str(base.get('agent_id')))}'s first "
                       f"answer</h3><div class=wrap><table><tr>{''.join(cells)}</tr></table></div>")
            shown += 1
    return "".join(out)


def _median(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def takeaways(rows, traits, cells, bad, man, ratio, as_found, eff, g9_found, g9_q, p9_q, o_found, w_found, w_q,
              ch, sides, beyond, corr, allm, small, entangled, readable, with_effect, through, hits, stronger_img) -> list:
    """What the sections add up to, as interpretive sentences built from their numbers (html strings)."""
    take = []
    read_b = {t: man[t]["vector"]["b"] for t in traits}
    up = lambda t: as_found[(t, "amplify (c > 0)")]
    down = lambda t: as_found[(t, "suppress (c < 0)")]
    lopsided = [t for t in traits if up(t) and down(t) and math.isfinite(up(t)["b"]) and math.isfinite(down(t)["b"])
                and max(up(t)["b"], down(t)["b"]) > 2 * max(min(up(t)["b"], down(t)["b"]), 1e-9)]
    moved = [t for t in traits if sig(man[t]["vector"]) and man[t]["vector"]["b"] > 0]
    take.append(
        f"<b>The vectors steer what agents say.</b> {len(moved)} of {len(traits)} traits move in the steered direction "
        f"(median {_median(read_b.values()):.1f} SD of the trait's readout per unit c); a random direction of the same "
        f"size moves it {100 * _median(ratio.values()):.0f}% as much (median), so the effect is the persona direction, "
        "not just any perturbation."
        + (" It is not symmetric: " + "; ".join(
            f"{t} is easier to {'remove' if down(t)['b'] > up(t)['b'] else 'add'} ({up(t)['b']:.1f} SD per unit c up, "
            f"{down(t)['b']:.1f} down)" for t in lopsided) + "." if lopsided else ""))

    rew = {t: eff[t]["pooled"] for t in traits}
    n_g = sum(1 for v in g9_found.values() if v and math.isfinite(v.get("p", float("nan"))))
    n_gs = sum(1 for k, v in g9_found.items() if sig(v, g9_q[k]))
    pooled_sig = [t for t in traits if sig(rew[t], p9_q[("all games", t)])]
    helps = [f"{t} ({rew[t]['b']:+.2f})" for t in pooled_sig if rew[t]["b"] > 0]
    hurts = [f"{t} ({rew[t]['b']:+.2f})" for t in pooled_sig if rew[t]["b"] < 0]
    biggest = max([abs(v["b"]) for v in rew.values() if v and math.isfinite(v["b"])] or [float("nan")])
    take.append(
        f"<b>Behaviour moves far less than language.</b> Per unit c the readout shifts by a median "
        f"{_median([abs(b) for b in read_b.values()]):.1f} SD, the steered agent's own reward by at most {biggest:.2f} SD "
        f"(pooled); {n_gs} of {n_g} game × trait reward slopes are significant. "
        + (f"More of the trait pays off for {', '.join(helps)}" if helps else "")
        + ("; it " if helps and hurts else "") + (f"costs the agent for {', '.join(hurts)}" if hurts else "")
        + (" (SD per unit c). " if helps or hurts else "")
        + "A persona that is plainly visible in the text often leaves the moves unchanged.")

    by_cond = defaultdict(list)
    for (c, _), v in o_found.items():
        if v and math.isfinite(v["b"]):
            by_cond[c].append(abs(v["b"]))
    strong = sorted(((c, _median(v)) for c, v in by_cond.items()), key=lambda kv: -kv[1])
    if strong:
        w_neg = sum(1 for k, v in w_found.items() if sig(v, w_q[k]) and v["b"] < 0)
        w_pos = sum(1 for k, v in w_found.items() if sig(v, w_q[k]) and v["b"] > 0)
        top = strong[0][0]
        head = ("Steering the deceiving side moves outcomes most" if top.startswith("evil") else
                "Steering many agents moves outcomes most" if top in ("all", "half") else
                f"The setting '{top}' moves outcomes most")
        take.append(
            f"<b>{head}.</b> Typical size of the steered agents' reward effect per setting (median |slope| over the "
            "traits): " + ", ".join(f"{c} {v:.2f}" for c, v in strong) + " SD per unit c"
            + (" — differences between settings are small next to the language effects. "
               if strong[0][1] < 2 * strong[-1][1] else ". ")
            + (f"Group welfare (everyone's total) falls in {w_neg} and rises in {w_pos} significant setting × trait "
               "cells: steered groups mostly end up worse off together." if w_neg > w_pos else
               f"Group welfare (everyone's total): {w_neg} significant falls, {w_pos} rises."))

    ceiling = []
    for cell in cells:
        ev = [r["reward"] for r in rows if r["cell"] == cell and r["condition"] == "none" and r["team"] == "evil"
              and r["reward"] is not None]
        if ev and np.mean(ev) >= 0.8:
            ceiling.append(f"{cell_name(cell)} ({100 * np.mean(ev):.0f}%)")
    if ceiling:
        take.append(
            "<b>Hidden-role results hit a ceiling.</b> The deceiving side already wins most unsteered games in "
            f"{', '.join(ceiling)}, so steering can mostly only make it lose: a positive evil-side slope means 'loses "
            "when pushed away from the trait'. Whether a persona makes deceivers <i>better</i> needs a balanced "
            "baseline (e.g. more good players or stronger good-side rules).")

    if sides:
        weak_neg = [t for t, s in sides if s < 0 and math.isfinite(ch[t][s]["shift"][0]) and ch[t][s]["shift"][0] > -1]
        rp = corr.get("reward", {}).get("prompt_vector")
        rv = corr.get("reward", {}).get("vector_vector")
        take.append(
            "<b>A persona prompt is a close but not exact substitute for the vector.</b> Both shift the monitored "
            f"traits in a similar pattern (median cosine {_median(ch[t][s]['cos_vec'] for t, s in sides):.2f}); the "
            f"prompt is often as strong as the strongest vector or stronger ({len(beyond)} of {len(sides)} trait × "
            "sign cases beyond it)"
            + (f", but telling the model <i>not</i> to show a trait barely works ({', '.join(weak_neg)}: under 1 SD), "
               "where the vector suppresses it" if weak_neg else "")
            + (f". In play they agree only partly (reward effects r = {rp[0]:.2f}; two vector strengths agree "
               f"r = {rv[0]:.2f})" if rp and rv else "")
            + ". Only the vector has a strength knob, so dose–response questions need it.")

    if allm:
        take.append(
            "<b>A monitor sees <i>that</i> an agent is steered, not <i>which</i> trait.</b> The readout singles out "
            f"the steered agent (mean AUC {np.mean(allm):.2f}"
            + (f", {np.mean(small):.2f} already at |c| = 0.5" if small else "") + f"), but {entangled} of "
            f"{len(traits) * (len(readable) - 1)} cross-trait effects are significant: every vector moves the other "
            "traits too. Flagging the steered agent works; naming its persona from the readout alone does not.")

    if with_effect:
        silent = [t for t in with_effect if t not in through]
        tracked = [t for t in with_effect if t in through]
        take.append(
            ("<b>Some outcome changes are invisible in the language.</b> " if silent else
             "<b>Where outcomes change, the language goes along.</b> ")
            + f"Steering changes the agent's reward for {', '.join(with_effect)}. "
            + (f"For {', '.join(tracked)} agents whose readout moved more also changed their reward more "
               "(correlational; section 11b shows how much of the effect that accounts for). " if tracked else "")
            + (f"For {', '.join(silent)} the reward changes without the readout predicting it: behaviour (votes, bids, "
               "moves) shifts without the language showing why — the hard case for a text-based monitor." if silent else ""))

    if hits:
        take.append(
            f"<b>Personas are mildly contagious.</b> Unsteered co-players drift toward the steered trait "
            f"({'; '.join(hits[:3])} SD per unit c): steering one agent changes how the others talk too, so the "
            "unsteered agents of a steered episode are not a clean control.")

    if stronger_img:
        take.append(
            f"<b>Image observations amplify steering.</b> For {len(stronger_img)} of {len(traits)} traits the language "
            "moves more when the agent sees the game as a picture than as text.")

    unsteered = {c: len({r["cluster"] for r in rows if r["cell"] == c and r["condition"] == "none"}) for c in cells}
    thin = [c for c in cells if 0 < unsteered[c] < 20]
    none0 = [cell_name(c) for c in cells if not unsteered[c]]
    caveats = []
    if thin:
        caveats.append(f"fewer than 20 unsteered episodes in {len(thin)} of {len(cells)} games (median "
                       f"{_median(unsteered[c] for c in thin):.0f}); every paired effect leans on them, so more "
                       "unsteered episodes are the cheapest way to firmer numbers")
    if none0:
        caveats.append(f"no unsteered episodes yet in {', '.join(none0)}")
    if bad:
        caveats.append(f"over 5% of answers unparseable in {', '.join(bad)} (the agent then plays a default move), "
                       "so reward effects there partly reflect parsing")
    if caveats:
        take.append("<b>Read with care:</b> " + "; ".join(caveats) + ".")
    return take


def build(rows, sweep=None, logs_dir=None) -> str:
    rows = zscored(rows)
    traits = sorted({r["trait"] for r in rows if r["trait"]})
    cells = sorted({r["cell"] for r in rows})
    base = baselines(rows)
    scale = reward_scale(rows)
    key, body = [], []  # key findings (top of the page) and the sections

    # 1. coverage
    tab = ["<div class=wrap><table><tr><th class=l>game</th><th>runs</th><th>episodes</th><th>unsteered episodes</th>"
           "<th>cut off at the turn limit</th><th>unparseable answers</th></tr>"]
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell]
        eps = {r["cluster"]: r["truncated"] for r in cr}
        turns, fb = sum(r["turns"] for r in cr), sum(r["fallbacks"] for r in cr)
        tab.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td><td>{len({r['run_id'] for r in cr})}</td>"
                   f"<td>{len(eps)}</td><td>{len({r['cluster'] for r in cr if r['condition'] == 'none'})}</td>"
                   f"<td>{100 * sum(eps.values()) / max(len(eps), 1):.0f}%</td><td>{100 * fb / max(turns, 1):.1f}%</td></tr>")
    tab.append("</table></div>")
    bad = [cell_name(c) for c in cells
           if sum(r["fallbacks"] for r in rows if r["cell"] == c) > 0.05 * max(1, sum(r["turns"] for r in rows if r["cell"] == c))]
    body.append(section(1, "Coverage", "How much data is there per game, and did the games run cleanly?",
                        "Finished episodes per game; the share of episodes stopped at the turn limit; the share of "
                        "agent turns whose answer could not be parsed even after the retries (the agent then plays a "
                        "default move).", "Unsteered episodes are the reference for everything else: few of them make "
                        "all later numbers noisy. Unparseable answers above a few % point to a prompt/parser problem.",
                        [f"{len({r['cluster'] for r in rows})} episodes over {len(cells)} games."]
                        + ([f"More than 5% unparseable answers in: {', '.join(bad)}."] if bad else []), "".join(tab)))

    # 2. manipulation
    man = manipulation(rows, traits)
    q_man = bh([man[t]["vector"]["p"] for t in traits] + [man[t]["random"]["p"] for t in traits])
    tab = ["<div class=wrap><table><tr><th class=l>trait</th><th>vector slope</th><th>random-direction slope</th>"
           "<th>prompt + (mean)</th><th>prompt − (mean)</th></tr>"]
    for i, t in enumerate(traits):
        pp, pn = man[t]["prompt"][1], man[t]["prompt"][-1]
        tab.append(f"<tr><td class=l>{t}</td>{slope_cell(man[t]['vector'], q_man[i])}"
                   f"{slope_cell(man[t]['random'], q_man[len(traits) + i])}"
                   f"<td>{fmt(pp[0], 2)} ± {fmt(pp[1], 2)}</td><td>{fmt(pn[0], 2)} ± {fmt(pn[1], 2)}</td></tr>")
    tab.append("</table></div>" + LEGEND + "<div class=charts>")
    for t in traits:
        curve = man[t]["curve"]
        series = {"vector": [(x, m, h) for (kind, x), (m, h, _) in curve.items() if kind == "vector"],
                  "random": [(x, m, h) for (kind, x), (m, h, _) in curve.items() if kind == "random"]
                  + ([(0.0, *curve[("vector", 0.0)][:2])] if ("vector", 0.0) in curve else []),
                  "prompt": [(s, *man[t]["prompt"][s][:2]) for s in (1, -1)]}
        tab.append(f"<div class=chart><h3>{t}</h3>{line_chart(series, t, f'{t} readout (SD)')}</div>")
    tab.append("</div>")
    pg, pg_found, pg_q = matrix("Vector slope per game", "game", cells, traits, lambda c, t: man[t]["per_game"].get(c))
    moved = [t for i, t in enumerate(traits) if sig(man[t]["vector"], q_man[i]) and man[t]["vector"]["b"] > 0]
    ratio = {t: abs(man[t]["random"]["b"]) / abs(man[t]["vector"]["b"]) for t in traits
             if man[t]["vector"]["b"] and math.isfinite(man[t]["random"]["b"])}
    f2 = [f"<b>{len(moved)} of {len(traits)}</b> traits move significantly in the steered direction."
          + (f" Strongest: {ranked({t: man[t]['vector']['b'] for t in traits})}; weakest: "
             f"{ranked({t: man[t]['vector']['b'] for t in traits}, reverse=False)} SD per unit coefficient." if traits else "")]
    if ratio:
        f2.append(f"The random-direction control moves the readout by {100 * min(ratio.values()):.0f}–"
                  f"{100 * max(ratio.values()):.0f}% of the persona vector's effect: "
                  + ("the effect is mostly specific to the trait direction." if max(ratio.values()) < 0.5 else
                     "part of the effect is generic 'disturbance', not the trait."))
    neg = sum(1 for k, s in pg_found.items() if sig(s, pg_q[k]) and s["b"] < 0)
    f2.append(f"Per game: {sum(1 for k, s in pg_found.items() if sig(s, pg_q[k]) and s['b'] > 0)} of "
              f"{sum(1 for s in pg_found.values() if s and math.isfinite(s.get('b', float('nan'))))} game×trait cells "
              f"move significantly in the steered direction" + (f", {neg} in the opposite direction." if neg else "."))
    key.append(f2[0])
    body.append(section(2, "Does steering move the trait? (manipulation check)",
                        "When an agent is steered with trait X at coefficient c, does what it writes express more of X "
                        "(c &gt; 0) or less (c &lt; 0)?",
                        "The steered agent's own-trait readout (see Setup) per unit of coefficient. The unsteered agents "
                        "of the same games enter at c = 0. Controls: a random direction with the same norm as the "
                        "vector; the persona prompt (the instruction the vector was extracted with).",
                        "Charts: x = steering coefficient, y = readout of that trait in SD units above/below unsteered "
                        "agents. A rising red line = steering works and more coefficient gives more trait; the grey "
                        "dashed line should stay flat. Slope 0.5 means c = +2 reads about +1 SD.",
                        f2, "".join(tab) + pg))

    # 3. specificity (steered traits only)
    readable = [u for u in traits if any(u in r["mon"] for r in rows)]
    sp_html, sp_found, sp_q = matrix("", "steered ↓ / read →", traits, readable,
                                     lambda t, u: readout_slope(rows, t, lambda r: is_vector(r), readout=u))
    f3 = []
    for t in traits:
        own = sp_found.get((t, t))
        off = {u: sp_found[(t, u)]["b"] for u in readable if u != t and sig(sp_found[(t, u)], sp_q[(t, u)])}
        if own and math.isfinite(own.get("b", float("nan"))) and off:
            u, b = max(off.items(), key=lambda kv: abs(kv[1]))
            f3.append(f"<b>{t}</b> also moves {u} ({b:+.2f}, {100 * abs(b) / max(abs(own['b']), 1e-9):.0f}% of its own "
                      f"effect {own['b']:+.2f}).")
    entangled = sum(1 for (t, u), s in sp_found.items() if t != u and sig(s, sp_q[(t, u)]))
    f3.insert(0, f"{entangled} of {len(traits) * (len(readable) - 1)} off-diagonal cells are significant: "
                 + ("the persona vectors are strongly entangled (they share a common 'persona' axis), so a trait "
                    "readout alone does not tell which trait was steered." if entangled > len(traits) * 2 else
                    "the vectors are fairly specific."))
    key.append(f3[0])
    body.append(section(3, "Specificity: what else moves?",
                        "Steering trait X: which other traits' readouts change along with it?",
                        "For agents steered with the row trait, the slope of each column trait's readout per unit "
                        "coefficient (SD units). Only the 8 steered traits are shown as columns (the monitor reads all "
                        "PersVecFast traits; the CSV has them all).",
                        "The diagonal is the intended effect (as in section 2). Off-diagonal red/blue cells are side "
                        "effects: e.g. steering 'sarcastic' raising 'evil' means the two vectors overlap.",
                        f3, sp_html))

    # 4. detectability
    det = {t: detectability(rows, t) for t in traits}
    coefs = sorted({c for d in det.values() for c in d})
    tab = ["<div class=wrap><table><tr><th class=l>trait</th>" + "".join(f"<th>c = {c:g}</th>" for c in coefs) + "</tr>"]
    for t in traits:
        tds = []
        for c in coefs:
            m, h, n = det[t].get(c, (float("nan"), float("nan"), 0))
            shade = (f' style="background:rgba(192,57,43,{max(0.0, min(1.0, (m - 0.5) * 2)) * 0.55:.2f})"'
                     if math.isfinite(m) else "")
            tds.append(f'<td{shade} title="episodes={n}">{fmt(m, 2)}{"" if not math.isfinite(h) else f" ±{h:.2f}"}</td>')
        tab.append(f"<tr><td class=l>{t}</td>{''.join(tds)}</tr>")
    tab.append("</table></div>")
    allm = [m for d in det.values() for (m, _, _) in d.values() if math.isfinite(m)]
    small = [m for d in det.values() for c, (m, _, _) in d.items() if abs(c) <= 0.5 and math.isfinite(m)]
    f4 = []
    if allm:
        f4.append(f"Average AUC {np.mean(allm):.2f} (0.5 = chance). " + (
            "A readout monitor singles out the steered agent in most episodes." if np.mean(allm) > 0.75 else
            "The steered agent is only weakly visible to the monitor."))
    if small:
        f4.append(f"Even at |c| = 0.5 the AUC averages {np.mean(small):.2f}.")
    key.extend(f4[:1])
    body.append(section(4, "Detectability: can a monitor tell who is steered?",
                        "In an episode where one agent is steered, does the readout single it out among its co-players?",
                        "Per episode: the share of co-players whose readout of the steered trait is lower than the "
                        "steered agent's (for c &lt; 0: higher). Averaged over episodes per trait and coefficient (AUC).",
                        "0.5 = the monitor cannot tell; 1.0 = the steered agent always has the most extreme readout. "
                        "Darker red = more detectable.", f4, "".join(tab)))

    # 5. asymmetry
    as_html, as_found, as_q = matrix("", "trait", traits, ["amplify (c > 0)", "suppress (c < 0)"],
                                     lambda t, side: readout_slope(rows, t, lambda r: is_vector(r) and (
                                         (r["coef"] > 0) if side.startswith("amp") else (r["coef"] < 0))))
    f5 = []
    for t in traits:
        a, s_ = as_found[(t, "amplify (c > 0)")], as_found[(t, "suppress (c < 0)")]
        if a and s_ and math.isfinite(a.get("b", float("nan"))) and math.isfinite(s_.get("b", float("nan"))):
            if abs(a["b"]) > 1.5 * abs(s_["b"]):
                f5.append(f"{t}: amplifying is stronger ({a['b']:.2f} vs {s_['b']:.2f}).")
            elif abs(s_["b"]) > 1.5 * abs(a["b"]):
                f5.append(f"{t}: suppressing is stronger ({s_['b']:.2f} vs {a['b']:.2f}).")
    body.append(section(5, "Amplify vs suppress", "Is pushing toward a trait as effective as pushing away from it?",
                        "The section-2 slope fitted separately on the positive and on the negative coefficients (each "
                        "with the unsteered agents at 0).", "Equal values = symmetric. A trait the model rarely shows "
                        "unprompted has little to suppress (floor effect).",
                        f5 or ["Roughly symmetric for all traits."], as_html))

    # 6. persona prompt vs persona vector
    ch, agree, corr = channels(rows, traits, base, scale)
    sides = [(t, s) for t in traits if t in ch for s in (1, -1) if s in ch[t]]
    sgn = lambda s: "+" if s > 0 else "−"

    def eq_text(eq):
        x, note = eq
        return "–" if not math.isfinite(x) else f"beyond {x:+g}" if note == ">" else "≤ 0" if note == "<" else f"{x:+.1f}"

    # a. strength
    tab_a = ["<h3>a. Strength: how far does each move the trait's own readout?</h3><div class=wrap><table><tr>"
             "<th class=l>trait</th><th>prompt + (SD)</th><th>≈ vector c</th><th>vector at max + (SD)</th>"
             "<th>prompt − (SD)</th><th>≈ vector c</th><th>vector at max − (SD)</th>"
             "<th>vector-only slope</th><th>prompt + vector slope</th></tr>"]
    for t in traits:
        if t not in ch:
            continue
        both = readout_slope(rows, t, lambda r: r["steered"] and r["via"] == "both" and not r["random"])
        only = readout_slope(rows, t, lambda r: r["steered"] and r["via"] == "vector" and not r["random"])
        cols = "".join(f"{mean_cell(ch[t][s]['shift'])}<td>{eq_text(ch[t][s]['eq'])}</td><td>{fmt(ch[t][s]['vmax'], 2)}</td>"
                       if s in ch[t] else "<td class=muted>–</td>" * 3 for s in (1, -1))
        tab_a.append(f"<tr><td class=l>{t}</td>{cols}{slope_cell(only)}{slope_cell(both)}</tr>")
    tab_a.append("</table></div>")

    # b. direction
    tab_b = ["<h3>b. Direction: what else moves? (fingerprint over all monitored traits)</h3><div class=wrap><table><tr>"
             "<th class=l>trait</th><th>side</th><th>cos(prompt, vector)</th><th>cos(prompt, random)</th>"
             "<th class=l>moved most by the prompt (SD)</th><th class=l>moved most by the vector at max (SD)</th>"
             "<th>own trait's rank: prompt / vector</th></tr>"]
    top = lambda lst: ", ".join(f"{html.escape(u)} {v:+.2f}" for u, v in lst)
    for t, s in sides:
        d = ch[t][s]
        tab_b.append(f"<tr><td class=l>{t}</td><td>{sgn(s)}</td><td>{fmt(d['cos_vec'], 2)}</td><td>{fmt(d['cos_rnd'], 2)}</td>"
                     f"<td class=l>{top(d['top_p'])}</td><td class=l>{top(d['top_v'])}</td>"
                     f"<td>{d['rank_p'] or '–'} / {d['rank_v'] or '–'}</td></tr>")
    tab_b.append("</table></div>")

    # c. behaviour
    tab_c = ["<h3>c. Behaviour: what happens in the game?</h3>"
             "<div class=wrap><table><tr><th class=l>trait</th><th>side</th><th>own reward: prompt (one agent)</th>"
             "<th>own reward: vector at max (one agent)</th><th>own reward: vector, next-lower |c|</th>"
             "<th>evil side's reward: prompt (evil team)</th><th>evil side's reward: vector at max</th>"
             "<th>welfare: prompt (all agents)</th><th>welfare: vector at max (all agents)</th></tr>"]
    differ = []
    for t, s in sides:
        d = ch[t][s]
        tab_c.append(f"<tr><td class=l>{t}</td><td>{sgn(s)}</td>{mean_cell(d['reward']['prompt'])}{mean_cell(d['reward']['vector'])}"
                     f"{mean_cell(d['reward']['vector_next'])}{mean_cell(d['evil']['prompt'])}{mean_cell(d['evil']['vector'])}"
                     f"{mean_cell(d['welfare']['prompt'])}{mean_cell(d['welfare']['vector'])}</tr>")
        for what, who in (("reward", "steered agent"), ("evil", "evil side")):
            a, b = d[what]["prompt"], d[what]["vector"]
            if all(math.isfinite(v) for v in (a[0], a[1], b[0], b[1])) and abs(a[0] - b[0]) > math.hypot(a[1], b[1]):
                differ.append(f"{t} {sgn(s)}, {who} (prompt {a[0]:+.2f}, vector {b[0]:+.2f})")
    tab_c.append("</table></div>")
    charts = "<div class=charts>"
    for k, what in (("readout", "trait readout (SD)"), ("reward", "own reward (SD)")):
        r_pv, r_vv = corr.get(k, {}).get("prompt_vector"), corr.get(k, {}).get("vector_vector")
        cap = (f"r = {fmt(r_pv[0], 2)} [{fmt(r_pv[1], 2)}, {fmt(r_pv[2], 2)}], n = {r_pv[3]}; reference, vector at "
               f"max vs next-lower |c|: r = {fmt(r_vv[0], 2)}" if r_pv else "")
        charts += (f"<div class=chart><h3>Per game: {what}</h3>"
                   + scatter_chart([(b, a, lab) for a, b, _, lab in agree.get(k, [])], f"vector at max: {what}",
                                   f"prompt: {what}") + f"<span class=muted>{cap}</span></div>")
    charts += ("</div><div class=legend><span><i style='background:var(--pos);height:7px;width:7px'></i>+ side "
               "(instruction / c &gt; 0)</span><span><i style='background:var(--neg);height:7px;width:7px'></i>− side"
               "</span><span>dashed: y = x (same effect)</span><span>one dot per trait × game; hover for names</span></div>")

    # d. visibility and cost
    tab_d = ["<h3>d. Visibility and cost</h3><div class=wrap><table><tr><th class=l>trait</th><th>side</th>"
             "<th>detectability AUC: prompt</th><th>AUC: vector at max</th><th>AUC: prompt + vector at max</th>"
             "<th>co-players' readout: prompt</th><th>co-players': vector at max</th>"
             "<th>answer length: prompt / vector / unsteered</th><th>unparseable %: prompt / vector</th></tr>"]
    auc_p, auc_v, len_p, len_v, len_0 = [], [], [], [], []
    for t, s in sides:
        d = ch[t][s]
        cp, cv, c0 = d["cost"]["prompt"], d["cost"]["vector"], d["cost"]["none"]
        tab_d.append(f"<tr><td class=l>{t}</td><td>{sgn(s)}</td>{mean_cell(d['det']['prompt'])}{mean_cell(d['det']['vector'])}"
                     f"{mean_cell(d['det']['both'])}{mean_cell(d['contagion']['prompt'])}{mean_cell(d['contagion']['vector'])}"
                     f"<td>{fmt(cp[0], 0)} / {fmt(cv[0], 0)} / {fmt(c0[0], 0)}</td><td>{fmt(cp[1], 1)} / {fmt(cv[1], 1)}</td></tr>")
        if d["det"]["prompt"] and d["det"]["vector"]:
            auc_p.append(d["det"]["prompt"][0])
            auc_v.append(d["det"]["vector"][0])
        len_p.append(cp[0])
        len_v.append(cv[0])
        len_0.append(c0[0])
    tab_d.append("</table></div>")

    f6 = []
    within = [(t, s) for t, s in sides if not ch[t][s]["eq"][1] and math.isfinite(ch[t][s]["eq"][0])]
    beyond = [f"{t} {sgn(s)}" for t, s in sides if ch[t][s]["eq"][1] == ">"]
    if sides:
        f6.append(f"<b>Strength:</b> for {len(within)} of {len(sides)} trait × side cases the prompt's effect on the "
                  f"readout equals the vector's somewhere in the tested range (median |c| ≈ "
                  f"{np.median([abs(ch[t][s]['eq'][0]) for t, s in within]) if within else float('nan'):.1f})"
                  + (f"; the prompt goes beyond the strongest vector for {', '.join(beyond)}." if beyond else "."))
        cv_ = [ch[t][s]["cos_vec"] for t, s in sides if math.isfinite(ch[t][s]["cos_vec"])]
        cr_ = [ch[t][s]["cos_rnd"] for t, s in sides if math.isfinite(ch[t][s]["cos_rnd"])]
        if cv_:
            f6.append(f"<b>Direction:</b> the prompt and the vector move the 59 monitored traits in a similar pattern "
                      f"(mean cosine {np.mean(cv_):.2f}, range {min(cv_):.2f}–{max(cv_):.2f}"
                      + (f"; prompt vs random direction: {np.mean(cr_):.2f}" if cr_ else "") + "). "
                      + ("They act as largely the same intervention on the language." if np.mean(cv_) > 0.7 else
                         "Similar in part: each also moves traits the other does not (table b)."
                         if np.mean(cv_) > 0.4 else "They change the language in different ways."))
        rp = corr.get("reward", {}).get("prompt_vector")
        rv = corr.get("reward", {}).get("vector_vector")
        if rp and math.isfinite(rp[0]):
            f6.append(f"<b>Behaviour:</b> per game, the prompt's effect on the agents' reward correlates r = {rp[0]:.2f} "
                      f"with the vector's (two vector strengths agree r = {fmt(rv[0], 2)}, the ceiling given the noise). "
                      + (f"Clearly different pooled reward effects: {'; '.join(differ)}." if differ else
                         "No trait × side where the pooled reward effects clearly differ."))
        if auc_p:
            f6.append(f"<b>Visibility:</b> the monitor singles out a prompted agent with mean AUC {np.mean(auc_p):.2f}, a "
                      f"vector-steered one (max c) with {np.mean(auc_v):.2f}.")
        f6.append(f"<b>Cost:</b> mean answer length {np.nanmean(len_p):.0f} characters with the prompt, "
                  f"{np.nanmean(len_v):.0f} with the vector at max, {np.nanmean(len_0):.0f} unsteered.")
        key.append(f6[0].replace("<b>Strength:</b>", "<b>Prompt vs vector:</b>"))
    body.append(section(6, "Persona prompt vs persona vector: can the prompt stand in for the vector?",
                        "Is writing the persona into the system prompt comparable to adding the persona vector — as "
                        "strong, in the same direction, with the same effect on play, as visible, at the same cost?",
                        "The same trait given as the PersVecFast instruction in the system prompt (+ = express it, "
                        "− = its opposite; no strength knob) vs as the vector at c = ±0.5..±2. "
                        "<b>a</b>: mean readout of the trait (SD vs unsteered) and the vector coefficient with the same "
                        "readout (interpolated on the vector's dose-response curve). <b>b</b>: the shift of all 59 "
                        "monitored traits as a profile; cosine 1 = same pattern, 0 = unrelated. <b>c</b>: paired reward "
                        "(same seat, same seed, unsteered; SD units) of the one steered agent (prompt_one vs one), of "
                        "the deceiving side when all its members are steered (evil_team_prompt vs evil_team), and the "
                        "group's total when every agent is steered (prompt_all vs all; ~0 in zero-sum games by "
                        "construction); scatter: one dot per trait × side × game. <b>d</b>: AUC as in section 4 and contagion as in 8 (one agent "
                        "steered); answer length and unparseable answers.",
                        "≈ vector c tells which vector strength the prompt corresponds to ('beyond' = stronger than "
                        "every tested c). In the scatters, dots on the dashed diagonal = same effect; compare the "
                        "prompt–vector r with the reference r (two vector strengths), which shows how well two "
                        "near-identical interventions can agree given the noise. Only games run with both channels.",
                        f6, "".join(tab_a) + "".join(tab_b) + "".join(tab_c) + charts + "".join(tab_d)))

    # 7. who is steered
    conds = sorted({r["condition"] for r in rows if is_vector(r)})
    r7, f7_read, _ = matrix("Readout of the steered agents (SD per unit c)", "condition", conds, traits,
                            lambda c, t: readout_slope(rows, t, lambda r: is_vector(r) and r["condition"] == c))
    own = paired(rows, base, lambda r: is_vector(r))
    o7, o_found, o_q = matrix("Steered agents' own reward (SD per unit c)", "condition", conds, traits,
                              lambda c, t: _diff_slope([r for r in own if r["condition"] == c and r["trait"] == t], scale))
    wel = welfare([r for r in rows if r["condition"] != "none"], [r for r in rows if r["condition"] == "none"])
    w7, w_found, w_q = matrix("Group welfare: everyone's total reward (SD per unit c)", "condition", conds, traits,
                              lambda c, t: _diff_slope([r for r in wel if r["condition"] == c and r["trait"] == t], None))
    f7 = [f"Own reward, significant: {'; '.join(sig_cells(o_found, o_q, lambda k: f'{k[0]} × {k[1]}')[:6]) or 'none'}.",
          f"Group welfare, significant: {'; '.join(sig_cells(w_found, w_q, lambda k: f'{k[0]} × {k[1]}')[:6]) or 'none'}."]
    key.append(f7[1])
    body.append(section(7, "Who is steered: one, half, all, one side",
                        "Does it matter how many agents are steered, and on which side? Who gains, who loses?",
                        "Per condition × trait, three slopes per unit coefficient: the steered agents' readout; their "
                        "own reward vs the same seat in the unsteered episode with the same seed; the total reward of "
                        "<i>all</i> agents in the episode vs the unsteered episode with the same seed.",
                        "Rows are conditions (see Setup). Positive welfare = steering makes the group as a whole better "
                        "off. In zero-sum games (Avalon: one side wins) welfare stays ~0 by construction.",
                        f7, r7 + o7 + w7))

    # 8. contagion
    partial = sorted({r["condition"] for r in rows if is_vector(r) and r["select"] in ("one", "count", "fraction")})
    c8, c_found, c_q = matrix("", "condition", partial, traits,
                              lambda c, t: readout_slope(rows, t, lambda r: r["condition"] == c and not r["steered"]))
    hits = sig_cells(c_found, c_q, lambda k: f"{k[1]} ({k[0]})")
    f8 = [("The unsteered co-players pick up the trait for: " + "; ".join(hits[:6]) + ".") if hits else
          "No significant contagion: the co-players' language does not shift with the steered agent's trait."]
    key.append(f8[0])
    body.append(section(8, "Contagion: do the others start to sound like the steered agent?",
                        "When one agent (or half) is steered, does the trait spread to the agents that are NOT steered?",
                        "The unsteered co-players' readout of the steered trait, per unit of the steered agent's "
                        "coefficient (SD units), compared with agents in unsteered episodes.",
                        "Positive = the trait spreads through the conversation (or the others answer in kind); "
                        "negative = the others move the opposite way (e.g. calming a sarcastic player).", f8, c8))

    # 9-10. outcomes
    steered = paired(rows, base, lambda r: is_vector(r) and r["team_cond"] is None)
    eff = effect_table(steered, traits, scale)
    p9, p9_found, p9_q = matrix("All games pooled (SD per unit c)", "", ["all games"], traits, lambda _, t: eff[t]["pooled"])
    g9, g9_found, g9_q = matrix("Per game (raw reward per unit c)", "game", cells, traits, lambda c, t: eff[t].get(c))
    pooled_hits = sig_cells(p9_found, p9_q, lambda k: k[1])
    f9 = [("Pooled over games, the steered agent's reward changes for: " + "; ".join(pooled_hits) + ".") if pooled_hits
          else "Pooled over games, steering does not significantly change the steered agent's own reward.",
          "Per game, significant: " + ("; ".join(sig_cells(g9_found, g9_q, lambda k: f"{k[1]} in {cell_name(k[0])}")[:8])
                                       or "none") + "."]
    key.append(f9[0])
    body.append(section(9, "Outcome: does the steered agent do better or worse?",
                        "Does steering an agent change how well it does in the game?",
                        "The steered agent's reward minus the reward of the same seat in the unsteered episode with the "
                        "same seed (same deal, same role), regressed on the coefficient. Pooled: rewards in SD units of "
                        "each game's unsteered rewards.", "Positive = more of the trait helps the agent; the sign flips "
                        "for suppression. Per-game rows are in the game's own reward units.", f9, p9 + g9))
    spill = paired(rows, base, lambda r: not r["steered"] and r["trait"] and r["select"] in ("one", "count", "fraction")
                   and r["team_cond"] is None and r["via"] in STEERED_VIA and not r["random"])
    eff_s = effect_table(spill, traits, scale)
    p10, p10_found, p10_q = matrix("All games pooled (SD per unit c)", "", ["all games"], traits,
                                   lambda _, t: eff_s[t]["pooled"])
    g10, g10_found, g10_q = matrix("Per game (raw reward per unit c)", "game", cells, traits, lambda c, t: eff_s[t].get(c))
    sp_hits = sig_cells(p10_found, p10_q, lambda k: k[1])
    body.append(section(10, "Outcome: what happens to the others?",
                        "When one agent (or half) is steered, do the unsteered agents do better or worse?",
                        "As in 9, for the UNsteered agents of those episodes.",
                        "Negative = the steered agent's trait costs its co-players.",
                        [("Co-players' reward changes for: " + "; ".join(sp_hits) + ".") if sp_hits else
                         "No significant effect on the co-players' reward, pooled."], p10 + g10))

    # 11. evil team
    team_rows = paired(rows, base, lambda r: r["team_cond"] == "evil" and r["team"] == "evil" and is_vector(r))
    if team_rows:
        eff_t = effect_table(team_rows, traits, scale)
        p11, p11_found, p11_q = matrix("Pooled (SD per unit c)", "", ["hidden-role games"], traits,
                                       lambda _, t: eff_t[t]["pooled"])
        tcells = sorted({r["cell"] for r in team_rows})
        g11, _, _ = matrix("Per game (raw reward per unit c)", "game", tcells, traits, lambda c, t: eff_t[t].get(c))
        th = sig_cells(p11_found, p11_q, lambda k: k[1])
        f11 = [("Steering the whole deceiving side changes its win rate for: " + "; ".join(th)
                + " (positive = the deceivers win more).") if th else "No significant effect on the deceiving side's reward."]
        key.append(f11[0])
        body.append(section(11, "Steering the deceiving side (hidden-role games)",
                            "If the whole evil team (spies, werewolves, mafia) is steered, does it win more or less?",
                            "The evil agents' reward vs the same seats in the unsteered episode with the same seed, per "
                            "unit coefficient.", "Avalon is zero-sum: the evil side's gain is the good side's loss.",
                            f11, p11 + g11))

    # 11b. link
    lk = {t: link(steered, t, scale) for t in traits}
    q_lk = bh([lk[t]["total"]["p"] for t in traits] + [lk[t]["assoc"]["p"] for t in traits]
              + [lk[t]["direct"]["p"] for t in traits])
    n_t = len(traits)
    tab = ["<div class=wrap><table><tr><th class=l>trait</th><th>coefficient → outcome</th><th>readout → outcome</th>"
           "<th>coefficient → outcome, readout held fixed</th><th>share through the readout</th></tr>"]
    for i, t in enumerate(traits):
        sh = lk[t]["share"]
        ok = sig(lk[t]["total"], q_lk[i]) and sig(lk[t]["via_readout"]) and math.isfinite(sh) and 0 <= sh <= 1
        tab.append(f"<tr><td class=l>{t}</td>{slope_cell(lk[t]['total'], q_lk[i])}"
                   f"{slope_cell(lk[t]['assoc'], q_lk[n_t + i])}{slope_cell(lk[t]['direct'], q_lk[2 * n_t + i])}"
                   f"<td>{f'{100 * sh:.0f}%' if ok else '<span class=muted>–</span>'}</td></tr>")
    tab.append("</table></div>")
    with_effect = [t for i, t in enumerate(traits) if sig(lk[t]["total"], q_lk[i])]
    through = [t for i, t in enumerate(traits) if sig(lk[t]["assoc"], q_lk[n_t + i])]
    f11b = [f"Steering changes the outcome for: {', '.join(with_effect) or 'no trait'}; the readout predicts the outcome "
            f"for: {', '.join(through) or 'no trait'}."]
    if with_effect and not set(with_effect) & set(through):
        f11b.append("Where steering changes outcomes, it does <b>not</b> go through the measured language: behaviour "
                    "(votes, bids, moves) changes without the readout showing why — the hard case for a monitor.")
    key.append(f11b[-1])
    body.append(section("11b", "Does the trait in the language go with the outcome?",
                        "Are the agents whose language changed most also the ones whose outcome changed?",
                        "Steered agents only. Coefficient → outcome: total effect (as in 9). Readout → outcome: reward "
                        "change per +1 SD of readout. Readout held fixed: the steering effect the readout does not "
                        "account for. Share: 1 − fixed/total, shown only when both paths are significant.",
                        "Correlational: it shows whether outcome changes go along with language changes, not that the "
                        "language causes them.", f11b, "".join(tab)))

    # 12. text vs image
    obs_read, obs_rew, both_modes = {}, {}, []
    for t in traits:
        obs_read[t], both_modes = obs_compare(rows, t, lambda sub, t: readout_slope(sub, t, lambda r: is_vector(r)))
        obs_rew[t], _ = obs_compare(steered, t, lambda sub, t: _diff_slope([r for r in sub if r["trait"] == t], scale))
    i12, i_found, i_q = matrix("Readout slope (SD per unit c)", "trait", traits, ["text", "image", "diff"],
                               lambda t, m: obs_read[t][m])
    j12, _, _ = matrix("Steered agent's reward slope (SD per unit c)", "trait", traits, ["text", "image", "diff"],
                       lambda t, m: obs_rew[t][m])
    stronger_img = [t for t in traits if sig(i_found[(t, "diff")], i_q[(t, "diff")]) and i_found[(t, "diff")]["b"] > 0]
    weaker_img = [t for t in traits if sig(i_found[(t, "diff")], i_q[(t, "diff")]) and i_found[(t, "diff")]["b"] < 0]
    f12 = [f"Games played in both modes: {', '.join(both_modes) or 'none yet'}."]
    if stronger_img:
        f12.append(f"Steering moves the language <b>more</b> when the agent sees an image, for {len(stronger_img)} of "
                   f"{len(traits)} traits ({', '.join(stronger_img)}).")
    if weaker_img:
        f12.append(f"Less with images for: {', '.join(weaker_img)}.")
    if stronger_img or weaker_img:
        key.append(f12[1])
    body.append(section(12, "Text vs image observations",
                        "Does steering act differently when the agent sees the game as a picture instead of text?",
                        "Only games played in both modes. The readout slope (section 2) and the steered agent's reward "
                        "slope (section 9), separately for text and image runs; diff = image − text.",
                        "A significant diff means the observation format changes how strongly the persona takes hold.",
                        f12, i12 + j12))

    # 13. coherence
    co = [r for r in rows if r["turns"] and (is_vector(r) or r["condition"] == "none")]
    fb, ln = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    for r in co:
        x = r["coef"] if r["condition"] != "none" else 0.0
        fb[x][0].append(100 * r["fallbacks"] / r["turns"])
        fb[x][1].append(r["cluster"])
        if r["chars"] is not None:
            ln[x][0].append(r["chars"])
            ln[x][1].append(r["cluster"])
    fb_c = {x: mean_ci(*v) for x, v in fb.items()}
    ln_c = {x: mean_ci(*v) for x, v in ln.items()}
    charts = ("<div class=charts>"
              f"<div class=chart><h3>Unparseable answers</h3>"
              f"{line_chart({'vector': [(x, m, h) for x, (m, h, _) in fb_c.items()]}, 'fallbacks', '% of turns')}</div>"
              f"<div class=chart><h3>Answer length</h3>"
              f"{line_chart({'vector': [(x, m, h) for x, (m, h, _) in ln_c.items()]}, 'length', 'characters')}</div></div>")
    f13 = []
    if 0.0 in ln_c and ln_c:
        hi_x, lo_x = max(ln_c), min(ln_c)
        f13.append(f"Answers are {ln_c[0.0][0]:.0f} characters long unsteered, {ln_c[hi_x][0]:.0f} at c = {hi_x:g} and "
                   f"{ln_c[lo_x][0]:.0f} at c = {lo_x:g}.")
    if fb_c:
        steered = {x: v[0] for x, v in fb_c.items() if x != 0.0 and math.isfinite(v[0])}
        if 0.0 in fb_c and steered:
            worst = max(steered, key=steered.get)
            f13.append(f"Unparseable answers: {fb_c[0.0][0]:.1f}% of turns unsteered, {min(steered.values()):.1f}–"
                       f"{steered[worst]:.1f}% steered (most at c = {worst:g})"
                       + (" — on average steering does not break answers more than the games' own parse failures."
                          if steered[worst] <= fb_c[0.0][0] + 2 else " — strong steering breaks answers."))
        # pooled numbers hide that it depends on game and trait: the cells far from their game's unsteered rate
        per = defaultdict(list)
        for r in co:
            key = (r["cell"], None) if r["condition"] == "none" else (r["cell"], r["trait"], r["coef"])
            per[key].append(r["fallbacks"] / r["turns"])
        base_fb = {k[0]: statistics.mean(v) for k, v in per.items() if k[1] is None}
        cells_fb = [(k, statistics.mean(v)) for k, v in per.items() if k[1] is not None and k[0] in base_fb and len(v) >= 20]
        broken = sorted((x for x in cells_fb if x[1] - base_fb[x[0][0]] >= 0.25), key=lambda x: -x[1])
        fixed = sorted((x for x in cells_fb if base_fb[x[0][0]] - x[1] >= 0.2), key=lambda x: x[1] - base_fb[x[0][0]])
        if broken:
            f13.append("Steering breaks answers (≥ 25 points above the game's unsteered rate) in: " + "; ".join(
                f"{cell_name(k[0])} × {k[1]} c = {k[2]:g}: {100 * v:.0f}% (unsteered {100 * base_fb[k[0]]:.0f}%)"
                for k, v in broken[:8]) + (f"; {len(broken) - 8} more" if len(broken) > 8 else "") + ".")
        if fixed:
            f13.append("Steering leaves fewer answers unparseable than unsteered (≥ 20 points) in: " + "; ".join(
                f"{cell_name(k[0])} × {k[1]} c = {k[2]:g}: {100 * v:.0f}% (unsteered {100 * base_fb[k[0]]:.0f}%)"
                for k, v in fixed[:6]) + (f"; {len(fixed) - 6} more" if len(fixed) > 6 else "")
                + ". Where a game's reward depends on parsing, part of a reward effect there can come from this.")
    body.append(section(13, "Coherence cost: does steering break the agent?",
                        "Does strong steering make agents produce broken or unusually long/short answers?",
                        "Steered agents (persona vector, all traits) by coefficient, unsteered agents at 0: the share of "
                        "turns that stayed unparseable after the retries, and the mean answer length.",
                        "x = coefficient; a curve that shoots up at the ends = the model derails before the persona shows.",
                        f13, charts))

    body.append(games_section(rows, traits, cells, man, eff, base, scale))

    # takeaways: what the sections add up to, written from their numbers
    take = takeaways(rows, traits, cells, bad, man, ratio, as_found, eff, g9_found, g9_q, p9_q, o_found, w_found, w_q,
                     ch, sides, beyond, corr, allm, small, entangled, readable, with_effect, through, hits, stronger_img)
    take_html = ("<h2 id=takeaways>Takeaways</h2><div class=result><ol>" + "".join(f"<li>{t}</li>" for t in take)
                + "</ol><span class=muted>What the sections add up to, written from their numbers; the evidence for "
                  "each point is in the sections below.</span></div>")

    head = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width, "
            f"initial-scale=1'><title>Steering report</title><style>{CSS}</style></head><body><h1>Steering report</h1>"
            f"<p class=muted>Persona-vector steering of LLM agents in multi-agent games: does it change what they say, "
            f"how they play, and what happens to the group?</p>"
            f"<div class=result><b>Key findings</b><ul>{''.join(f'<li>{k}</li>' for k in key)}</ul>"
            f"<span class=muted>Generated from the numbers below; each section explains its measure.</span></div>")
    toc = ("<p class=muted>Sections: <a href=#takeaways>takeaways</a> · <a href=#setup>setup</a> · <a href=#examples>examples</a> · " + " · ".join(
        f"<a href=#s{n}>{n}</a>" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "11b", 12, 13)) + " · <a href=#games>per game</a></p>")
    coefs = sorted({r["coef"] for r in rows if r["coef"] is not None and r["via"] != "prompt"})
    ex = examples_section(logs_dir, traits, coefs) if logs_dir else ""
    return head + toc + take_html + setup_section(rows, sweep, traits, cells) + ex + "".join(body) + "</body></html>"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("agents_csv", help="output of scripts/summarize_runs.py")
    p.add_argument("--out", required=True)
    p.add_argument("--sweep", help="the sweep file (model, generation settings) for the setup section")
    p.add_argument("--logs", help="the sweep's log dir (<logging_dir>/<name>): adds matched example answers")
    args = p.parse_args(argv)
    rows = load(args.agents_csv)
    if not rows:
        print("no rows")
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sweep = __import__("yaml").safe_load(Path(args.sweep).read_text(encoding="utf-8")) if args.sweep else None
    Path(args.out).write_text(build(rows, sweep, args.logs), encoding="utf-8")
    print(f"report over {len(rows)} agent-episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

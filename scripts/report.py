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
  14 over the episode the readout per third of the agent's turns: does the persona fade, does contagion build
  15 actions         numeric choices and action shares (votes, bids, play/discard/hint) per unit c
  16 dose-response   the reward curve over c per game x trait: rising, saturating, peaked; per setting: is the
                     bend real (lack of fit vs a line), pushed down vs up, mild vs strong
  17 footprint       each vector's effect on all monitored personas, not only its own trait
  18 naming          can a classifier name the injected trait from the readout (held-out episodes)
  19 vector/prompt/both side by side at c = ±1: readout, reward, cost, contagion, per game, groups
  20 length          readout slope with answer length held fixed
  21 group           group size vs one agent's effect; everyone steered vs one steered (welfare)
  22 hidden roles    the deceivers' win rate by who carries the persona, across games (up vs down,
                     strong vs mild, team vs one member vs an honest player)
  23 natural         unsteered agents' trait-reward correlation vs the steered effect
  24 risk, tempo     spread of reward changes by |c|; game length when everyone is steered
  25 image x channel per game x trait slopes with images vs text (language, result), the curves with their
                     fitted lines; vector / prompt / both per mode; synergy both - (vector + prompt) per mode
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
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_runs import csv_parts  # noqa: E402

STEERED_VIA = ("vector", "both")


# ---------------------------------------------------------------- data

def read_csv(path):
    """The rows of summarize_runs' output `path`, over all its parts (agents.csv.gz, agents-2.csv.gz, ...)."""
    for part in csv_parts(path):
        with (gzip.open if str(part).endswith(".gz") else open)(part, "rt", newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)


def load(path) -> list:
    rows = []
    for r in read_csv(path):
        num = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
        row = {
            "run_id": r["run_id"], "group": r.get("group") or r["run_id"].split("__")[0],
            "condition": r.get("condition") or (r["run_id"].split("__")[1] if "__" in r["run_id"] else "none"),
            "obs": r.get("observation") or "text", "episode": int(r["episode"]), "agent": r["agent"],
            "trait": r.get("condition_trait") or None, "coef": num("condition_coefficient"),
            "via": r.get("condition_via") or None, "random": r.get("condition_random") == "True",
            # a random direction at a fixed layer (all_random_L16, ...) rather than at the persona's own: its own
            # setting, kept out of the random control the persona's vector is compared with
            "random_fixed": r.get("condition_random") == "True" and r.get("condition_layer") not in (None, ""),
            "team_cond": r.get("condition_team") or None, "select": r.get("condition_select") or None,
            "steered": r.get("steered") == "True", "team": r.get("team") or None,
            "reward": num("reward"), "reward_from": r.get("reward_from") or None,
            "turns": num("turns") or 0, "fallbacks": num("parse_fallbacks") or 0,
            "tokens_in": num("tokens_in"), "tokens_out": num("tokens_out"),
            "truncated": r.get("truncated") == "True", "chars": num("answer_chars"),
            "players": num("players"), "episode_turns": num("episode_turns"),
            "monitor_layer": num("monitor_layer"), "steer_layer": num("steer_layer"),
            "mon": {k[4:]: float(v) for k, v in r.items() if k.startswith("mon_") and v not in (None, "")
                    and not k.startswith(("mon_early_", "mon_mid_", "mon_late_"))},
            "phase": {ph: {k[len(ph) + 5:]: float(v) for k, v in r.items()
                           if k.startswith(f"mon_{ph}_") and v not in (None, "")} for ph in ("early", "mid", "late")},
            "act_n": num("act_n"), "act_num": num("act_numeric"),
            "act_labels": json.loads(r["act_labels"]) if r.get("act_labels") else {},
        }
        rows.append(row)
    for r in rows:
        r["cell"] = (r["group"], r["obs"])  # a game as played: group and observation mode
        r["cluster"] = (r["run_id"], r["episode"])
    return rows


def role_cells(rows) -> set:
    """Game cells with hidden roles (a deceiving side): there the group's total reward only says which side won."""
    return {r["cell"] for r in rows if r["team"] == "evil"}


def last_step_only(rows) -> list:
    """Games that pay every step but whose reward here is only the last step's (CSV built before episode totals)."""
    try:
        from scripts.backfill_rewards import PER_STEP_GAMES
    except ImportError:
        return []
    return sorted({cell_name(r["cell"]) for r in rows if r["group"].rsplit("_", 1)[0] in PER_STEP_GAMES
                   and r["reward_from"] in (None, "last_step") and r["reward"] is not None})


def baselines(rows) -> dict:
    """{(group, obs, episode, agent): reward} of the unsteered runs: same seed = same deal and roles."""
    return {(r["group"], r["obs"], r["episode"], r["agent"]): r["reward"]
            for r in rows if r["condition"] == "none" and r["reward"] is not None}


# ---------------------------------------------------------------- statistics

def slope(x, y, clusters, fe=None) -> dict:
    """OLS slope of y on x (with fixed effects `fe` if given), SE clustered by `clusters`. With game cells as `fe`,
    the result also says which games went into it ("games"), for the note under each table."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    clusters = [c for c, k in zip(clusters, keep) if k]
    n = len(x)
    if fe is not None:
        fe = [f for f, k in zip(fe, keep) if k]
    games = {"games": sorted(set(fe))} if fe and all(isinstance(f, tuple) and len(f) == 2 for f in fe) else {}
    if n < 3:
        return {"n": n, "b": float("nan"), "se": float("nan"), "p": float("nan"), **games}
    if fe is not None:
        x, y = _demean(x, fe), _demean(y, fe)
    else:
        x, y = x - x.mean(), y - y.mean()
    sxx = float((x * x).sum())
    if sxx <= 0:
        return {"n": n, "b": float("nan"), "se": float("nan"), "p": float("nan"), **games}
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
    return {"n": n, "clusters": k, "b": b, "se": se, "p": p, **games}


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


def chi2_sf(x, k):
    """P(X > x) for a chi-square variable with k (integer) degrees of freedom."""
    if k < 1 or not math.isfinite(x):
        return float("nan")
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        term = total = 1.0
        for j in range(1, k // 2):
            term *= x / 2 / j
            total += term
        return min(1.0, math.exp(-x / 2) * total)
    total, term = math.erfc(math.sqrt(x / 2)), math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for j in range(1, (k - 1) // 2 + 1):
        term *= x / (2 * j - 1) if j > 1 else 1.0
        total += term
    return min(1.0, total)


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
        rnd = [r for r in rows if r["steered"] and r["trait"] == t and r["random"] and not r["random_fixed"]
               and t in r["mon"]]
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
            for ph, vs in r["phase"].items():  # each third against the same third of unsteered agents
                for t, v in vs.items():
                    vals[(r["cell"], ph, t)].append(v)
    stats = {k: (float(np.mean(v)), float(np.std(v)) or 1.0) for k, v in vals.items() if len(v) > 1}
    out = []
    for r in rows:
        z = {t: (v - stats[(r["cell"], t)][0]) / stats[(r["cell"], t)][1]
             for t, v in r["mon"].items() if (r["cell"], t) in stats}
        zp = {ph: {t: (v - stats[(r["cell"], ph, t)][0]) / stats[(r["cell"], ph, t)][1]
                   for t, v in vs.items() if (r["cell"], ph, t) in stats} for ph, vs in r["phase"].items()}
        out.append({**r, "mon_raw": r["mon"], "mon": z, "phase": zp})
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


def ols(y, X, clusters, fe=None, width=2):
    """Multiple regression y ~ X (columns, with game fixed effects `fe`); coefficients with SEs clustered by
    `clusters`. Returns [{"b", "se", "p", "n"}] per column (`width` of them, all NaN, when there are no rows: e.g.
    readout terms of a sweep run without the monitor)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    if not len(y):
        X = np.empty((0, width))
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
    roles = role_cells(rows)  # their total reward only says which side won: not a welfare
    wel = welfare([r for r in rows if r["condition"] in ("all", "prompt_all") and r["cell"] not in roles],
                  [r for r in rows if r["condition"] == "none" and r["cell"] not in roles])
    out, agree = {}, defaultdict(list)  # agree: [(prompt, vector, vector at the next-lower |c|, label)] per measure
    for t in traits:
        vec = [r for r in rows if r["steered"] and r["trait"] == t and r["via"] == "vector" and not r["random"]
               and r["team_cond"] is None]
        pro = [r for r in rows if r["steered"] and r["trait"] == t and r["via"] == "prompt" and r["team_cond"] is None]
        cells = {r["cell"] for r in vec} & {r["cell"] for r in pro}
        vec, pro = [r for r in vec if r["cell"] in cells], [r for r in pro if r["cell"] in cells]
        rnd = [r for r in rows if r["steered"] and r["trait"] == t and r["random"] and not r["random_fixed"]
               and r["cell"] in cells]
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
  --rnd:#8a8a90; --warn:#fff4d6; --ok:#e9f6ec;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#0e8a9e; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececee; --muted:#9a9aa1; --line:#2c2c30;
  --card:#1e1e21; --warn:#3a3220; --ok:#1d3326;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#33aec2; } }
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
.charts.mini { grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:8px; }
details.bysetting { margin:10px 0 4px; border-left:3px solid var(--line); padding-left:10px; }
details.bysetting > summary { color:var(--muted); }
.charts.mini .chart h3 { font-size:13px; }
.chart { background:var(--card); border-radius:8px; padding:8px; }
.chart h3 { margin:0 0 2px; }
svg text { fill:var(--muted); font-size:10px; }
.sig { font-weight:700; }
.games { color:var(--muted); font-size:12.5px; margin:-8px 0 14px; max-width:1000px; }
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


def line_chart(series, title, ylabel, xlabel="steering coefficient", width=300, height=200, colours=None, notes=None,
               span=None, fits=None):
    """series: {name: [(x, mean, half_ci)]} -> SVG with axes, labels and a zero line. `notes`: {series: text} shown
    when hovering that series' line (e.g. its summary); `span`: (low, high) of the y axis, shared by charts shown
    side by side; `fits`: {series: (intercept, slope)} drawn as a dashed straight line over the x range (the fitted
    slope, so the slope is seen and not only read off a table)."""
    pts = [p for s in series.values() for p in s if math.isfinite(p[1])]
    if not pts:
        return "<p class=muted>no data yet</p>"
    xs = [p[0] for p in pts]
    fits = {k: v for k, v in (fits or {}).items() if v and all(map(math.isfinite, v))}
    ends = [a + b * x for a, b in fits.values() for x in (min(xs), max(xs))]
    lo = min([p[1] - (p[2] if math.isfinite(p[2]) else 0) for p in pts] + [0] + ends)
    hi = max([p[1] + (p[2] if math.isfinite(p[2]) else 0) for p in pts] + [0] + ends)
    lo, hi = (lo - 1, hi + 1) if hi == lo else (lo, hi)
    lo, hi = span if span and span[1] > span[0] else (lo, hi)
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    L, R, T, B = 44, 8, 8, 34
    sx = lambda x: L + (x - x0) / (x1 - x0) * (width - L - R)
    sy = lambda y: T + (hi - y) / (hi - lo) * (height - T - B)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{height - B}" stroke="currentColor" stroke-opacity=".3"/>',
             f'<line x1="{L}" y1="{sy(0):.1f}" x2="{width - R}" y2="{sy(0):.1f}" stroke="currentColor" '
             f'stroke-opacity=".3" stroke-dasharray="2 2"/>']
    for v in (lo, 0, hi):
        parts.append(f'<text x="{L - 4}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:.3g}</text>')
    for v in sorted(set(xs)):
        parts.append(f'<text x="{sx(v):.1f}" y="{height - B + 12}" text-anchor="middle">{v:g}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text transform="translate(10,{(T + height - B) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 f'{html.escape(ylabel)}</text>')
    colours = {"vector": "var(--pos)", "random": "var(--rnd)", "prompt": "var(--neg)", **(colours or {})}
    for name, (a, b) in fits.items():
        parts.append(f'<line x1="{sx(x0):.1f}" y1="{sy(a + b * x0):.1f}" x2="{sx(x1):.1f}" y2="{sy(a + b * x1):.1f}" '
                     f'stroke="{colours.get(name, "currentColor")}" stroke-width="1.2" stroke-opacity=".7" '
                     f'stroke-dasharray="5 3"><title>{html.escape(name)}: fitted slope {b:+.2f} per unit c</title></line>')
    for name, s in series.items():
        s = sorted(p for p in s if math.isfinite(p[1]))
        c = colours.get(name, "currentColor")
        square = "prompt" in name  # prompts have no strength: points at ±1, no line
        dash = ' stroke-dasharray="4 3"' if "random" in name else ""
        if len(s) > 1 and not square:
            pts_ = " ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m, _ in s)
            parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.8"{dash} points="{pts_}"/>')
            if notes and name in notes:  # a wide invisible copy: an easy hover target with the line's note
                parts.append(f'<polyline fill="none" stroke="{c}" stroke-opacity="0" stroke-width="9" points="{pts_}">'
                             f'<title>{html.escape(notes[name])}</title></polyline>')
        for x, m, h in s:
            if math.isfinite(h):
                parts.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(m - h):.1f}" y2="{sy(m + h):.1f}" '
                             f'stroke="{c}" stroke-opacity=".6"/>')
            shape = (f'<rect x="{sx(x) - 3.5:.1f}" y="{sy(m) - 3.5:.1f}" width="7" height="7" fill="{c}">'
                     if square else f'<circle cx="{sx(x):.1f}" cy="{sy(m):.1f}" r="2.8" fill="{c}">')
            parts.append(shape + f'<title>{name}, c={x:g}: {m:.2f} ± {h:.2f}</title>'
                         + ("</rect>" if square else "</circle>"))
    parts.append("</svg>")
    return "".join(parts)


def scatter_chart(points, xlabel, ylabel, width=320, height=280, colour=None):
    """points: [(x, y, label)] -> SVG scatter with zero lines and the diagonal y = x (same effect both ways).
    `colour`: label -> CSS colour of its dot (default: red for labels with '+', blue otherwise)."""
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
        parts.append(f'<text x="{L - 4}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:.3g}</text>')
        parts.append(f'<text x="{sx(v):.1f}" y="{height - B + 12}" text-anchor="middle">{v:.2g}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text transform="translate(10,{(T + height - B) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 f'{html.escape(ylabel)}</text>')
    for x, y, lab in pts:
        c = colour(lab) if colour else "var(--pos)" if "+" in lab else "var(--neg)"
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{c}" fill-opacity=".75">'
                     f'<title>{html.escape(lab)}: x={x:.2f}, y={y:.2f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def forest_chart(items, colours, xlabel, width=330, span=None):
    """items: [(row label, {series: (mean, half CI)})] -> SVG with one row per item and, per series, a dot with its
    95% CI (series offset a little vertically, coloured by `colours` {series: CSS colour}), and a zero line.
    `span`: (low, high) of the x axis, shared by charts shown side by side; values beyond it are drawn at the edge."""
    vals = [(m - (h if math.isfinite(h) else 0), m + (h if math.isfinite(h) else 0))
            for _, ser in items for m, h in ser.values() if math.isfinite(m)]
    if not vals:
        return "<p class=muted>no data yet</p>"
    lo, hi = min([a for a, _ in vals] + [0]), max([b for _, b in vals] + [0])
    lo, hi = span if span and span[1] > span[0] else (lo, hi)
    lo, hi = (lo - 1, hi + 1) if hi == lo else (lo, hi)
    L, R, T, B, row = 104, 10, 6, 32, 17
    height = T + B + row * len(items)
    sx = lambda v: L + (min(max(v, lo), hi) - lo) / (hi - lo) * (width - L - R)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(xlabel)}">',
             f'<line x1="{sx(0):.1f}" y1="{T}" x2="{sx(0):.1f}" y2="{height - B}" stroke="currentColor" '
             f'stroke-opacity=".35" stroke-dasharray="2 2"/>']
    for v in (lo, 0, hi):
        parts.append(f'<text x="{sx(v):.1f}" y="{height - B + 12}" text-anchor="middle">{v:.3g}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    names = list(colours)
    for i, (lab, ser) in enumerate(items):
        y0 = T + row * i + row / 2
        if i % 2:
            parts.append(f'<rect x="{L}" y="{T + row * i}" width="{width - L - R}" height="{row}" fill="currentColor" '
                         f'fill-opacity=".04"/>')
        parts.append(f'<text x="{L - 6}" y="{y0 + 3:.1f}" text-anchor="end">{html.escape(lab)}</text>')
        for j, name in enumerate(names):
            m, h = ser.get(name, (float("nan"), float("nan")))
            if not math.isfinite(m):
                continue
            y, c = y0 + (j - (len(names) - 1) / 2) * 4, colours[name]
            if math.isfinite(h):
                parts.append(f'<line x1="{sx(m - h):.1f}" x2="{sx(m + h):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" '
                             f'stroke-opacity=".6"/>')
            parts.append(f'<circle cx="{sx(m):.1f}" cy="{y:.1f}" r="2.8" fill="{c}"><title>{html.escape(lab)}, '
                         f'{html.escape(name)}: {m:.2f} ± {h:.2f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


LEGEND = ('<div class=legend><span><i style="background:var(--pos)"></i>persona vector</span>'
          '<span><i style="background:var(--rnd)"></i>random direction, same norm (control)</span>'
          '<span><i style="background:var(--neg);height:7px;width:7px"></i>persona prompt (+ / − instruction)</span>'
          '<span>bars: 95% CI</span></div>')


def cell_name(cell):
    group, obs = cell
    return group if obs == "text" else f"{group} ({obs})"


CELL_NAMES = set()  # every game of the report as cell_name() writes it (set by build), for section_games


def section_games(body) -> list:
    """The games a section's numbers come from: those its games notes list, and those it shows by name (a per-game
    row, chart title or panel)."""
    found = set()
    for note in re.findall(r"<p class=games>(.*?)</p>", body, re.S):
        for tok in re.split(r"[;,]", re.sub(r"<b>.*?</b>", "", note)):
            tok = html.unescape(re.sub(r"^\s*\(\d+\):?\s*", "", tok).strip())
            if tok in CELL_NAMES:
                found.add(tok)
    found |= {n for n in CELL_NAMES if f">{html.escape(n)}<" in body}
    return sorted(found)


def section(num, title, question, measured, read, findings, body, warn=None):
    """One analysis: what it asks, what exactly is measured, how to read it, what it shows, the numbers; and which
    games its results come from."""
    items = "".join(f"<li>{f}</li>" for f in findings) or "<li>not enough data yet</li>"
    games = section_games(body)
    played = (f"<b>Games played for these results</b><span>{len(games)}: {html.escape(', '.join(games))}</span>"
              if games else "")
    return (f"<h2 id=s{num}>{num}. {html.escape(title)}</h2>"
            f"<div class=box><b>Question</b><span>{question}</span><b>What is measured</b><span>{measured}</span>"
            f"{played}<b>How to read it</b><span>{read}</span></div>"
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


def games_note(cells, what="Games in this table"):
    """'Games in this table (n): a, b (image), ...' under a table: which games (game × observation mode) its
    numbers come from."""
    cells = sorted(set(cells))
    if not cells:
        return ""
    return (f"<p class=games><b>{html.escape(what)} ({len(cells)}):</b> "
            f"{', '.join(html.escape(cell_name(c)) for c in cells)}</p>")


def matrix(title, corner, row_keys, col_keys, get, games=None):
    """(html, {key: slope}, {key: q}) for a rows x columns table of slopes, BH-corrected across its cells. Under it:
    the games its numbers come from (`games`, else gathered from the slopes and from rows/columns that are games)."""
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
    if games is None:
        games = {g for s in found.values() if s and math.isfinite(s.get("b", float("nan"))) for g in s.get("games", ())}
        games |= {c for (a, b), s in found.items() if s and math.isfinite(s.get("b", float("nan")))
                  for c in (a, b) if isinstance(c, tuple)}
    if not (len(games) == 1 and cell_name(next(iter(games))) == title):  # a per-game table names its game already
        out.append(games_note(games))
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
        if r.get("steer_layer") is not None and r["trait"] and not r.get("random_fixed"):  # a fixed-layer control
            layers[r["trait"]].add(int(r["steer_layer"]))
    mon_layer = sorted({r["monitor_layer"] for r in rows if r.get("monitor_layer") is not None})
    sw = sweep or {}
    cond_help = {
        "none": "nobody steered (the baseline every comparison is made against)",
        "one": "one randomly drawn agent gets the persona vector", "half": "half of the agents (drawn at random)",
        "all": "every agent", "all_random": "every agent gets a random direction of the same norm (control)",
        "all_random_L16": "as all_random, at layer 16 for every persona", "all_random_L32": "as all_random, at layer 32 "
        "for every persona",
        "random_one": "one agent gets a random direction of the same norm (control)",
        "prompt_one": "one agent gets the trait's persona instruction in its system prompt (no vector)",
        "prompt_all": "every agent gets the persona instruction", "both_one": "one agent gets the instruction AND the vector",
        "evil_team": "every member of the deceiving side (hidden-role games)", "evil_one": "one member of the deceiving side",
        "good_one": "one member of the honest side", "good_team": "every member of the honest side (hidden-role games)",
        "evil_team_prompt": "the deceiving side gets the persona instruction",
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
           "<b>Monitor (readout): how it works</b><span>"
           + (monitor_how(len({t for r in rows for t in r["mon"]}), mon_layer) if mon_layer else
              "<b>Not run for these episodes</b> (played without the monitor engine). Every readout result (sections "
              "2–4, 8, 11b, 14, 17, 18, 20, 23 and the readout columns elsewhere) is empty until the monitor is run "
              "over the logged turns; rewards, actions, coherence and length do not need it.") + "</span>",
           "<b>Outcome</b><span>Each game's own reward for the agent (e.g. win/loss of its side in Avalon, Werewolf, "
           "Mafia; points or goods won in auctions, harvest, TextArena games). Compared <i>paired</i>: the same seat in "
           "the unsteered episode with the same seed (same deal, same role).</span>",
           "<b>Statistics</b><span>Effects are slopes per unit coefficient (a straight line through the condition's "
           "points and the unsteered agents at 0), with games as fixed effects when pooled. Standard errors are "
           "clustered by episode. <b>*</b> q&lt;.05, <b>**</b> q&lt;.01, <b>***</b> q&lt;.001 after Benjamini–Hochberg "
           "correction over all cells of a table. Cell colour: red = increase, blue = decrease, stronger = more "
           "certain. Hover a cell for n, SE, p, q; '–' = not enough data.</span></div>"]
    last = last_step_only(rows)
    if last:
        out.append("<div class=warn><b>Rewards of " + html.escape(", ".join(last)) + " are only their last step's "
                   "payoff</b>, not the episode's score: these games pay every step, and this CSV was built before "
                   "the episode totals (scripts/backfill_rewards.py, then summarize_runs.py). Every reward result for "
                   "them (sections 7, 9, 10, 16, 19, 21, 23, 24, per game) is provisional until the report is rebuilt."
                   "</div>")
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


SETTING_ORDER = ("one", "half", "all", "both_one", "good_one", "evil_one", "evil_team", "good_team", "all_random",
                 "prompt_one", "prompt_all", "evil_team_prompt", "all_random_L16", "all_random_L32")

PARSE_GAP = 5.0  # percentage points from the game's unsteered rate that count as more (or fewer) unparseable turns
PARSE_MIN_TURNS = 50  # a case needs this many turns to be flagged


def unparseable_cases(rows, order=SETTING_ORDER) -> dict:
    """Unparseable turns of the steered agents per case (game cell, setting, coefficient), pooled over personas, and
    each game's base rate (its unsteered agents): {"base": {cell: %}, "cases": {(cell, setting, c): (%, turns,
    worst persona, its %)}}. Every setting counts, the random controls and prompts included."""
    rate = lambda rs: 100 * sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs))
    base = {c: rate(v) for c, v in _by([r for r in rows if r["condition"] == "none" and r["turns"]],
                                       lambda r: r["cell"]).items()}
    steered = [r for r in rows if r["steered"] and r["turns"] and r["trait"] and r["coef"] is not None
               and r["cell"] in base]
    per_trait = {k: rate(v) for k, v in _by(steered, lambda r: (r["cell"], r["condition"], r["coef"], r["trait"])).items()}
    cases = {}
    for (cell, cond, c), rs in _by(steered, lambda r: (r["cell"], r["condition"], r["coef"])).items():
        worst = max((t for (ce, co, cc, t) in per_trait if (ce, co, cc) == (cell, cond, c)),
                    key=lambda t: per_trait[(cell, cond, c, t)])
        cases[(cell, cond, c)] = (rate(rs), sum(r["turns"] for r in rs), worst, per_trait[(cell, cond, c, worst)])
    return {"base": base, "cases": cases, "order": order}


def _parse_flag(v, base, turns):
    """'more' | 'fewer' | '' for a case's rate against its game's base rate."""
    if turns < PARSE_MIN_TURNS:
        return ""
    return "more" if v >= base + PARSE_GAP else "fewer" if v <= base - PARSE_GAP else ""


def unparseable_tables(rows, order=SETTING_ORDER, words=None) -> str:
    """The cases of unparseable_cases as tables: a summary per game (base rate, how many cases are more or less
    unparseable, the worst case), then per game a setting x coefficient table with the base rate next to it. A case is
    flagged at PARSE_GAP points or more from the game's base rate (with at least PARSE_MIN_TURNS turns)."""
    u = unparseable_cases(rows, order)
    base, cases = u["base"], u["cases"]
    if not cases:
        return ""
    words = words or {}
    name = lambda c: html.escape(words.get(c, c))
    rank = lambda c: (order.index(c) if c in order else len(order), c)
    cstr = lambda cond, c: (("+ instruction" if c > 0 else "− instruction") if "prompt" in cond else f"c = {c:+g}")
    out = [f"<h3>Unparseable turns per case, against the game's base rate</h3><p class=muted>A case = game × setting × "
           f"strength; the steered agents' turns that stayed unparseable after the retries (the game then plays a "
           f"default move), pooled over personas; hover a cell for the persona with the most. Base rate: the same "
           f"game's unsteered agents. <span style='background:rgba(192,57,43,.35)'>&nbsp;more&nbsp;</span> = "
           f"{PARSE_GAP:g} points or more above the base rate, <span style='background:rgba(44,111,187,.35)'>&nbsp;fewer"
           f"&nbsp;</span> = {PARSE_GAP:g} or more below (cases with at least {PARSE_MIN_TURNS} turns). Where a game's "
           f"result depends on parsing (the default move can be good or bad), a reward effect in a flagged case is "
           f"partly a format effect.</p><div class=wrap><table><tr><th class=l>game</th><th>base rate (unsteered)</th>"
           f"<th>cases more unparseable</th><th>cases less unparseable</th><th class=l>worst case</th></tr>"]
    cells = sorted(base, key=lambda c: cell_name(c))
    for cell in cells:
        mine = {k: v for k, v in cases.items() if k[0] == cell}
        if not mine:
            continue
        flags = {k: _parse_flag(v[0], base[cell], v[1]) for k, v in mine.items()}
        (_, wc, wx), wv = max(mine.items(), key=lambda kv: kv[1][0])
        n_more, n_fewer = sum(f == "more" for f in flags.values()), sum(f == "fewer" for f in flags.values())
        out.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td><td>{base[cell]:.1f}%</td>"
                   f"<td>{n_more} of {len(mine)}</td><td>{n_fewer} of {len(mine)}</td>"
                   f"<td class=l>{name(wc)}, {cstr(wc, wx)}: <b>{wv[0]:.1f}%</b> (most: {html.escape(wv[2])}, "
                   f"{wv[3]:.1f}%)</td></tr>")
    out.append("</table></div>")
    no_base = sorted({cell_name(r["cell"]) for r in rows if r["steered"] and r["cell"] not in base})
    if no_base:
        out.append(f"<p class=muted>Not in these tables (no unsteered episodes yet, so no base rate): "
                   f"{html.escape(', '.join(no_base))}.</p>")
    colour = {"more": "192,57,43", "fewer": "44,111,187"}
    for cell in cells:
        mine = {k: v for k, v in cases.items() if k[0] == cell}
        if not mine:
            continue
        conds = sorted({k[1] for k in mine}, key=rank)
        coefs = sorted({k[2] for k in mine})
        tab = [f"<details class=bysetting><summary>{html.escape(cell_name(cell))}: base rate {base[cell]:.1f}%, "
               f"{sum(_parse_flag(v[0], base[cell], v[1]) == 'more' for v in mine.values())} case(s) more unparseable"
               f"</summary><div class=wrap><table><tr><th class=l>setting</th><th>base rate</th>"
               + "".join(f"<th>c = {c:+g}</th>" for c in coefs) + "</tr>"]
        for cond in conds:
            row = [f"<tr><td class=l>{name(cond)}</td><td class=muted>{base[cell]:.1f}%</td>"]
            for c in coefs:
                v = mine.get((cell, cond, c))
                if v is None:
                    row.append("<td class=muted>–</td>")
                    continue
                f = _parse_flag(v[0], base[cell], v[1])
                style = f' style="background:rgba({colour[f]},.35)"' if f else ""
                row.append(f'<td{style} title="{v[1]} turns; most: {html.escape(v[2])} {v[3]:.1f}%">{v[0]:.1f}%</td>')
            tab.append("".join(row) + "</tr>")
        tab.append("</table></div>" + ("<p class=muted>Prompt settings: c = −1 is the opposite instruction, +1 the "
                                       "instruction.</p>" if any("prompt" in c for c in conds) else "") + "</details>")
        out.append("".join(tab))
    return "".join(out)


def push_size(r, norms):
    """c times the norm of the vector the agent got at its layer (a random direction has the trait's vector norm,
    or the norm of PersVecFast's exported random file): how far the hidden state is moved. None without norms."""
    if not norms or r["coef"] is None or r["steer_layer"] is None or not r["trait"]:
        return None
    t, layer = r["trait"], str(int(r["steer_layer"]))
    n = (norms.get("random", {}).get(t, {}).get(layer) if r["random"] else None) or norms["norms"].get(t, {}).get(layer)
    return r["coef"] * n if n else None


def push_chart(series, fits, xlabel, ylabel, colours, width=330, height=240):
    """series: {name: [(x, mean, half CI, label)]} -> SVG with a continuous x axis, one colour per series and its
    fitted straight line `fits` {name: (intercept, slope)}, dashed."""
    pts = [p for s in series.values() for p in s if math.isfinite(p[1])]
    if len(pts) < 2:
        return "<p class=muted>no data yet</p>"
    xs = [p[0] for p in pts]
    x0, x1 = min(xs + [0]), max(xs + [0])
    ys = [p[1] + s * (p[2] if math.isfinite(p[2]) else 0) for p in pts for s in (-1, 1)]
    lo, hi = min(ys + [0]), max(ys + [0])
    lo, hi = (lo - 1, hi + 1) if hi == lo else (lo, hi)
    L, R, T, B = 44, 8, 8, 34
    sx = lambda v: L + (v - x0) / ((x1 - x0) or 1) * (width - L - R)
    sy = lambda v: T + (hi - v) / (hi - lo) * (height - T - B)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(ylabel)}">',
             f'<line x1="{L}" y1="{sy(0):.1f}" x2="{width - R}" y2="{sy(0):.1f}" stroke="currentColor" '
             f'stroke-opacity=".3" stroke-dasharray="2 2"/>',
             f'<line x1="{sx(0):.1f}" y1="{T}" x2="{sx(0):.1f}" y2="{height - B}" stroke="currentColor" stroke-opacity=".2"/>']
    for v in (lo, 0, hi):
        parts.append(f'<text x="{L - 4}" y="{sy(v) + 3:.1f}" text-anchor="end">{v:.3g}</text>')
    for v in (x0, 0, x1):
        parts.append(f'<text x="{sx(v):.1f}" y="{height - B + 12}" text-anchor="middle">{v:.3g}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(f'<text transform="translate(10,{(T + height - B) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 f'{html.escape(ylabel)}</text>')
    for name, fit in fits.items():
        s = [p[0] for p in series.get(name, [])]
        if fit and s and all(map(math.isfinite, fit)):
            a, b = fit
            parts.append(f'<line x1="{sx(min(s)):.1f}" y1="{sy(a + b * min(s)):.1f}" x2="{sx(max(s)):.1f}" '
                         f'y2="{sy(a + b * max(s)):.1f}" stroke="{colours[name]}" stroke-width="1.4" '
                         f'stroke-dasharray="5 3"><title>{html.escape(name)}: {b:+.3g} per unit push</title></line>')
    for name, s in series.items():
        for x, m, h, lab in s:
            if not math.isfinite(m):
                continue
            if math.isfinite(h):
                parts.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(m - h):.1f}" y2="{sy(m + h):.1f}" '
                             f'stroke="{colours[name]}" stroke-opacity=".5"/>')
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(m):.1f}" r="3" fill="{colours[name]}" fill-opacity=".8">'
                         f'<title>{html.escape(name)}, {html.escape(lab)}: push {x:.3g}, {m:.3g} ± {h:.2g}</title>'
                         f'</circle>')
    parts.append("</svg>")
    return "".join(parts)


def push_section(rows, base, scale, norms):
    """Section 26: the random directions' effect against how far they push (c · |v|), one line per layer, and the
    persona vectors per unit push next to the random direction at the same layer."""
    base_rate = {c: 100 * sum(r["fallbacks"] for r in v) / max(1, sum(r["turns"] for r in v))
                 for c, v in _by([r for r in rows if r["condition"] == "none" and r["turns"]], lambda r: r["cell"]).items()}
    use_norm = bool(norms)
    x_of = (lambda r: push_size(r, norms)) if use_norm else (lambda r: r["coef"])
    unit = "per unit push (c · |v|)" if use_norm else "per unit c"
    xlabel = "push c · |v| (signed)" if use_norm else "coefficient c (vector norms not found)"
    rnd = [dict(r, x=x_of(r)) for r in paired(rows, base, lambda r: r["steered"] and r["random"] and r["via"] != "prompt"
                                                and r["steer_layer"] is not None)]
    rnd = [r for r in rnd if r["x"] is not None]
    vec = [dict(r, x=x_of(r)) for r in paired(rows, base, lambda r: r["steered"] and r["condition"] == "all"
                                                and r["steer_layer"] is not None)]
    vec = [r for r in vec if r["x"] is not None]
    if not rnd:
        return [], ""
    layers = sorted({int(r["steer_layer"]) for r in rnd})
    colours = {f"L{l}": TRAIT_COLOURS[i % len(TRAIT_COLOURS)] for i, l in enumerate(layers)}
    measures = {"reward": ("steered agent's reward vs the same seat unsteered (SD)",
                           lambda r: r["diff"] / scale.get(r["cell"], 1.0)),
                "parse": ("unparseable turns, points above the game's unsteered rate",
                          lambda r: 100 * r["fallbacks"] / r["turns"] - base_rate[r["cell"]]
                          if r["turns"] and r["cell"] in base_rate else None)}
    num = lambda f: (f"{f['b']:+.3g}<span class=sig>{stars(f['p'])}</span>" if f and math.isfinite(f.get("b", float("nan")))
                     else "–")

    def block(rnd, vec):
        """charts (one line per layer) + the table rows (random vs the persona vectors at each layer) of one subset"""
        fits, charts = {}, []
        for m, (ylabel, y) in measures.items():
            series, lines = {}, {}
            for l in layers:
                rs = [r for r in rnd if int(r["steer_layer"]) == l and y(r) is not None]
                if not rs:
                    fits[(m, l)] = None
                    continue
                pts = []
                for (t, c), g in sorted(_by(rs, lambda r: (r["trait"], r["coef"])).items()):
                    mean, half, _ = mean_ci([y(r) for r in g], [r["cluster"] for r in g])
                    pts.append((float(np.mean([r["x"] for r in g])), mean, half, f"norm of {t}, c = {c:+g}"))
                series[f"L{l}"] = pts
                fits[(m, l)] = slope([r["x"] for r in rs], [y(r) for r in rs], [r["cluster"] for r in rs],
                                     [r["cell"] for r in rs])
                lines[f"L{l}"] = _line_fit([p[0] for p in pts], [p[1] for p in pts])
            charts.append(f"<div><h4>{html.escape(ylabel)}</h4>{push_chart(series, lines, xlabel, ylabel, colours)}</div>")
        trs = []
        for l in layers:
            rf, pf = fits[("reward", l)], fits[("parse", l)]
            per_t = []
            for t, g in sorted(_by([r for r in vec if int(r["steer_layer"]) == l], lambda r: r["trait"]).items()):
                s = slope([r["x"] for r in g], [r["diff"] / scale.get(r["cell"], 1.0) for r in g],
                          [r["cluster"] for r in g], [r["cell"] for r in g])
                d = _diff_test(rf, s) if rf else {"p": float("nan")}
                per_t.append(f"{html.escape(t)} {num(s)} (vs random: {'differs' if d['p'] < .01 else 'same' if d['p'] == d['p'] else '?'})")
            trs.append(f"<tr><td class=l>L{l}</td><td>{num(rf)}</td><td>{num(pf)}</td>"
                       f"<td class=l>{'; '.join(per_t) or '–'}</td></tr>")
        return fits, f"<div class=charts>{''.join(charts)}</div>", "".join(trs)

    legend = "".join(f"<span><i style='background:{c}'></i>random direction at {n}</span>" for n, c in colours.items())
    # the persona vectors (every agent steered) per unit push, beside the random direction at the same layer
    head = (f"<tr><th class=l>layer</th><th>random: reward {unit}</th><th>random: unparseable points {unit}</th>"
            f"<th class=l>persona vectors at this layer (every agent): reward {unit}</th></tr>")
    fits, charts, trs = block(rnd, vec)
    findings = []
    pv = lambda q: "p &lt; .001" if q < .001 else f"p = {q:.2g}"
    for l in layers:
        rf, pf = fits[("reward", l)], fits[("parse", l)]
        if rf and math.isfinite(rf["b"]):
            findings.append(f"Random direction at L{l}: reward {rf['b']:+.3g} {unit} ({pv(rf['p'])}), unparseable "
                            f"{pf['b']:+.3g} points {unit} ({pv(pf['p'])}).")
    # per game: the random slopes side by side, then each game's own charts and table
    cells = sorted({r["cell"] for r in rnd})
    g_head = ("<tr><th class=l>game</th>" + "".join(f"<th>L{l}: reward</th><th>L{l}: unparseable</th>" for l in layers)
              + "</tr>")
    g_rows, g_details, moved = [], [], {}
    for cell in cells:
        f, ch, tr = block([r for r in rnd if r["cell"] == cell], [r for r in vec if r["cell"] == cell])
        g_rows.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td>"
                      + "".join(f"<td>{num(f[('reward', l)])}</td><td>{num(f[('parse', l)])}</td>" for l in layers)
                      + "</tr>")
        g_details.append(f"<details><summary>{html.escape(cell_name(cell))}</summary>{ch}"
                         f"<div class=wrap><table>{head}{tr}</table></div></details>")
        for l in layers:
            rf = f[("reward", l)]
            if rf and math.isfinite(rf["b"]) and rf["p"] < .01:
                moved.setdefault(l, []).append(f"{cell_name(cell)} ({rf['b']:+.3g})")
    for l in layers:
        findings.append(f"Games where a random push at L{l} moves reward (p &lt; .01): "
                        f"{html.escape(', '.join(moved[l])) if l in moved else 'none'}.")
    per_game = (f"<h4>Per game: random-direction slopes {unit}</h4><div class=wrap><table>{g_head}{''.join(g_rows)}"
                f"</table></div>{''.join(g_details)}")
    body = (f"<div class=legend>{legend}<span>dots: mean per persona-norm × c, bars: 95% CI; dashed: straight line "
            f"through the dots</span></div>{charts}"
            f"<div class=wrap><table>{head}{trs}</table></div>" + per_game
            + games_note(cells, "Games in these charts"))
    return [], section(26, "Random directions: does the size of the push matter, per layer?",
                       "A random direction of a persona vector's size is the control: how much does pushing the hidden "
                       "state at all change the result, depending on how hard and at which layer? And do the persona "
                       "vectors do more than a random push of the same size at the same layer?",
                       "Every random-direction setting (at each persona's own layer and at fixed layers). x: the push, "
                       "c times the vector's norm at that layer (signed: −c is the opposite direction). y: the steered "
                       "agent's paired reward (SD units of the game) and its unparseable turns above the game's "
                       "unsteered rate. Slopes per layer pooled over games (game fixed effects, SE clustered by "
                       "episode). Table: the persona vectors' reward slope per unit push next to the random one at the "
                       "same layer ('differs' at p &lt; .01). Then the same per game (each game's own slopes, "
                       "charts and table).",
                       "A flat line = pushing at that layer does nothing by itself; a slope = any push changes the "
                       "result, so a persona vector there has to beat that line, not zero. Norms grow with depth, so "
                       "the same c is a bigger push at later layers.",
                       findings, body)


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


def turn_images(run_dir, episode, records, turn, agent_id):
    """The images `agent_id` saw at `turn` of `episode`, redrawn (the main sweep does not save images): the run's
    config (run.json) rebuilt with the episode's seed, then the logged actions of the earlier turns replayed. Shown
    only if the replay is faithful: the redrawn turn's text prompt must equal the logged one. PNG data URIs; []
    when the game cannot be rebuilt or replayed here."""
    try:
        import base64
        import io

        from testbed.config import RunConfig
        from testbed.registry import build_game
        from testbed.visual import fit, observe
        cfg = RunConfig.from_dict(json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8")))
        env, renderer, _ = build_game(family=cfg.game_family, game_id=cfg.game_id, num_players=cfg.players,
                                      env_kwargs={"seed": cfg.episode_seed(episode), **cfg.env_kwargs})
        env.reset()
        try:
            for t in sorted({r.get("turn", 0) for r in records if r.get("turn", 0) < turn}):
                env.submit({r["agent_id"]: r["parsed_action"] for r in records if r.get("turn", 0) == t})
            obs = dict(env.pending()).get(agent_id)
            if obs is None:
                return []
            text, images = observe(renderer, cfg.observation, obs, agent_id, getattr(env, "context", None))
        finally:
            getattr(env, "close", lambda: None)()
        rec = next(r for r in records if r.get("turn", 0) == turn and r["agent_id"] == agent_id)
        logged = (rec.get("attempts") or [{}])[0].get("user_prompt", rec.get("user_prompt"))
        if logged is not None and text != logged:
            return []
        out = []
        for im in images:
            buf = io.BytesIO()
            fit(im, cfg.image_size).save(buf, format="PNG")
            out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        return out
    except Exception:  # noqa: BLE001 - an example without its picture is still an example
        return []


def examples_section(logs_dirs, traits, coefs, per_trait=2, chars=700, later=4):
    """What steering looks like: the same seat's first turn of the same game (same seed, so the same prompt)
    unsteered, with the persona vector at the strongest - and + coefficient, the random direction and the
    persona prompt (+ / -); then the same seat's `later`-th answer, by which point each column's game has gone
    its own way. Conditions where every agent is steered (all, all_random, prompt_all), so the seat is steered in
    each. Per trait: `per_trait` text games (a different game for each trait where possible) and one game played
    from images, with the pictures the agent saw."""
    roots = [Path(d) for d in ([logs_dirs] if isinstance(logs_dirs, (str, Path)) else logs_dirs) if Path(d).is_dir()]
    if not roots or not coefs:
        return ""
    lo, hi = min(coefs), max(coefs)
    groups = sorted({p.name for r in roots for p in r.iterdir() if p.is_dir() and not p.name.startswith("_")})

    def run_dirs(g, run):
        return [r / g / run for r in roots if (r / g / run).is_dir()]

    def episodes(g, run):
        return {int(p.name.split("_")[1].split(".")[0]) for d in run_dirs(g, run) for p in d.glob("episode_*.summary.json")}

    def all_turns(g, run, ep):
        for d in run_dirs(g, run):
            turns = _turns(d / f"episode_{ep}.jsonl")
            if turns:
                return turns, d
        return [], None

    def answer(rec, t):
        mon = rec.get("monitor") or {}
        layer = sorted(mon, key=int)[len(mon) // 2] if mon else None
        score = mon.get(layer, {}).get(t) if layer else None
        text = (rec.get("completion") or "").strip()
        text = html.escape(text[:chars] + (" …" if len(text) > chars else "")).replace("\n", "<br>")
        return f"{text}<br><span class=muted>{t} readout: {fmt(score, 2)}</span>"

    def pictures(urls):
        return "".join(f"<img src='{u}' alt='what the agent saw' style='max-width:100%;display:block;margin:4px 0'>"
                       for u in urls) or "<span class=muted>(picture not redrawn)</span>"

    def example(t, g, suffix):
        runs = {"unsteered": f"{g}__none", f"vector c={lo:g}": f"{g}__all__{t}_x{lo:g}",
                f"vector c={hi:g}": f"{g}__all__{t}_x{hi:g}", f"random c={hi:g}": f"{g}__all_random__{t}_x{hi:g}",
                "prompt +": f"{g}__prompt_all__{t}_x1", "prompt −": f"{g}__prompt_all__{t}_x-1"}
        runs = {k: v + suffix for k, v in runs.items()}
        eps = None
        for run in runs.values():
            eps = episodes(g, run) if eps is None else eps & episodes(g, run)
        if not eps:
            return None
        ep = min(eps)
        logs = {k: all_turns(g, run, ep) for k, run in runs.items()}
        base = (logs["unsteered"][0] or [None])[0]
        if not base or len(base.get("completion") or "") < 120:  # one-word moves say little about style
            return None
        seat = base.get("agent_id")
        own = {k: [r for r in recs if r.get("agent_id") == seat] for k, (recs, _) in logs.items()}
        top = ("<td class=l style='vertical-align:top;min-width:220px'><b>{k}</b><br>{body}</td>")
        first = []
        for k, (recs, _) in logs.items():
            rec = recs[0] if recs else None
            first.append(top.format(k=k, body=answer(rec, t) if rec else "<span class=muted>missing</span>"))
        pics = ""
        if suffix:
            urls = turn_images(logs["unsteered"][1], ep, logs["unsteered"][0], base.get("turn", 0), seat)
            pics = ("<p class=muted>What the agent saw (same seed, so the same picture in every column):</p>"
                    + (f"<div style='max-width:360px'>{pictures(urls)}</div>" if urls else
                       "<p class=muted>(the picture could not be redrawn here)</p>"))
        n = min([len(v) for v in own.values()] + [later]) - 1  # the seat's n-th answer, in every column
        rows = f"<tr>{''.join(first)}</tr>"
        if n >= 1:
            cells = []
            for k, (recs, run_dir) in logs.items():
                rec = own[k][n]
                pic = pictures(turn_images(run_dir, ep, recs, rec.get("turn", 0), seat)) if suffix else ""
                cells.append(top.format(k=f"{k} · turn {rec.get('turn', '?')}", body=pic + answer(rec, t)))
            rows += (f"<tr><td class=l colspan={len(cells)}><span class=muted>Answer no. {n + 1} of the same seat: "
                     "from here each column's game has gone its own way, so the prompts (and pictures) differ."
                     "</span></td></tr><tr>" + "".join(cells) + "</tr>")
        mode = " (played from images)" if suffix else ""
        return (f"<h3>{t} — {html.escape(g)}{mode}, episode {ep}, {html.escape(str(seat))}'s first and a later "
                f"answer</h3>{pics}<div class=wrap><table>{rows}</table></div>")

    out = ["<h2 id=examples>What steering looks like: matched examples</h2><div class=box><b>What you see</b><span>"
           "For each trait, games where the same seat's <i>first</i> answer can be compared across conditions: same "
           "game, same seed, so the agent received exactly the same prompt; the only difference is the steering. "
           f"Columns: unsteered; persona vector at c = {lo:g} and {hi:g}; a random direction of the same norm at "
           f"c = {hi:g} (control); the persona prompt (+ / −). Second row: the same seat {later} answers in (or its "
           "last, if the game is shorter), when the games have diverged: does the persona hold, fade, or derail? "
           "The number under each answer is that turn's readout of the trait (raw projection, higher = more of the "
           "trait; compare within a row). Each trait shows different games where possible, and one game played from "
           "images with the pictures the agent saw (redrawn from the episode's seed and the logged moves; shown only "
           "when the redrawn prompt matches the logged one).</span><b>Why it matters</b><span>The tables below are "
           "averages over thousands of turns; these show what a slope of the readout actually corresponds to in text, "
           "including where strong steering stops being 'more of the trait' and becomes caricature or broken output."
           "</span></div>"]
    for i, t in enumerate(traits):
        order = groups[i * per_trait % max(len(groups), 1):] + groups[:i * per_trait % max(len(groups), 1)]
        for suffix, wanted in (("", per_trait), ("__obs-image", 1)):
            shown = 0
            for g in order:
                if shown >= wanted:
                    break
                ex = example(t, g, suffix)
                if ex:
                    out.append(ex)
                    shown += 1
    return "".join(out)


def _median(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def takeaways(rows, traits, cells, bad, man, ratio, as_found, eff, g9_found, g9_q, p9_q, o_found, w_found, w_q,
              ch, sides, beyond, corr, allm, small, entangled, readable, with_effect, through, hits, stronger_img,
              weaker_img) -> list:
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
        + "A persona that is plainly visible in the text often leaves the agent's result unchanged (its concrete "
          "choices can still shift: section 15).")

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
            + (f"Group welfare (everyone's total, games without hidden roles) falls in {w_neg} and rises in {w_pos} "
               "significant setting × trait cells: steered groups mostly end up worse off together." if w_neg > w_pos
               else f"Group welfare (everyone's total, games without hidden roles): {w_neg} significant falls, "
               f"{w_pos} rises."))

    high, low, balanced = [], [], []
    by_cell = _by(deceiver_wins([r for r in rows if r["condition"] == "none"]), lambda w: w["cell"])
    for cell, ws in sorted(by_cell.items()):
        rate = float(np.mean([w["won"] for w in ws]))
        (high if rate >= 80 else low if rate <= 20 else balanced).append(
            f"{cell_name(cell)} ({rate:.0f}%, n = {len(ws)})")
    if high or low:
        capped = [x.split(" (")[0] for x in high + low]
        take.append(f"<b>Hidden roles: {', '.join(capped)} {'is' if len(capped) == 1 else 'are'} near a ceiling"
                    + (", the others are not" if balanced else "") + ".</b> "
                    + (f"The deceivers win nearly every unsteered game in {', '.join(high)}, so there steering can "
                       "mostly only make them lose (a positive deceiver slope means 'loses when pushed away from the "
                       "trait')" if high else "")
                    + ("; " if high and low else "")
                    + (f"they almost never win in {', '.join(low)}, so there steering can mostly only help them"
                       if low else "")
                    + (f"; in {', '.join(balanced)} the sides are balanced, so effects can go either way: these are the "
                       "games that can show whether a persona makes deceivers <i>better</i>." if balanced else
                       ". Whether a persona makes deceivers better needs a balanced game."))

    if sides:
        weak_neg = [t for t, s in sides if s < 0 and math.isfinite(ch[t][s]["shift"][0]) and ch[t][s]["shift"][0] > -1]
        take.append(
            "<b>A persona prompt is a close but not exact substitute for the vector.</b> Both shift the monitored "
            f"traits in a similar pattern (median cosine {_median(ch[t][s]['cos_vec'] for t, s in sides):.2f}); the "
            f"prompt is often as strong as the strongest vector or stronger ({len(beyond)} of {len(sides)} trait × "
            "sign cases beyond it)"
            + (f", but telling the model <i>not</i> to show a trait barely works ({', '.join(weak_neg)}: under 1 SD), "
               "where the vector suppresses it" if weak_neg else "")
            + ". How alike they play is compared at matched strength in section 19 (takeaway 'Prompt, vector, "
              "both'). Only the vector has a strength knob, so dose–response questions need it.")

    if allm:
        take.append(
            "<b>A monitor sees <i>that</i> an agent is steered far better than <i>which</i> trait.</b> The readout "
            "singles out "
            f"the steered agent (mean AUC {np.mean(allm):.2f}"
            + (f", {np.mean(small):.2f} already at |c| = 0.5" if small else "") + f"), but {entangled} of "
            f"{len(traits) * (len(readable) - 1)} cross-trait effects are significant: every vector moves the other "
            "traits too, so one trait's readout does not say which trait was steered. Flagging the steered agent "
            "works; naming its persona from the whole readout profile works only partly (section 18).")

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

    if stronger_img or weaker_img:
        head = ("Image observations amplify steering." if len(stronger_img) > len(traits) / 2 and not weaker_img else
                "Images change how strongly steering takes hold.")
        take.append(
            f"<b>{head}</b> The language moves more when the agent sees the game as a picture than as text for "
            f"{len(stronger_img)} of {len(traits)} traits ({', '.join(stronger_img) or 'none'})"
            + (f", less for {', '.join(weaker_img)}" if weaker_img else "")
            + (f"; no clear difference for the other {len(traits) - len(stronger_img) - len(weaker_img)}"
               if len(stronger_img) + len(weaker_img) < len(traits) else "") + " (sections 12, 25).")

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
    last = last_step_only(rows)
    if last:
        caveats.append(f"rewards of {', '.join(last)} are only the last step's payoff in this data (they pay every "
                       "step), so their reward results are provisional until the report is rebuilt with episode totals")
    if caveats:
        take.append("<b>Read with care:</b> " + "; ".join(caveats) + ".")
    return take


PHASES = ("early", "mid", "late")


def _three(r, u):
    """The agent has a readout of `u` in all three thirds of its turns."""
    return all(u in (r["phase"].get(ph) or {}) for ph in PHASES)


def phase_slope(rows, t, select, value, readout=None):
    """Like readout_slope, on value(r, u) (a per-third readout) of agents with all three thirds."""
    u = readout or t
    chosen = [r for r in rows if r["trait"] == t and r["coef"] is not None and select(r) and _three(r, u)]
    cells = {r["cell"] for r in chosen}
    pts = chosen + [r for r in rows if r["condition"] == "none" and r["cell"] in cells and _three(r, u)]
    return slope([r["coef"] if r["condition"] != "none" else 0.0 for r in pts], [value(r, u) for r in pts],
                 [r["cluster"] for r in pts], [r["cell"] for r in pts])


def over_episode_section(rows, traits):
    """14: does the persona fade or grow over the episode; does contagion build up? -> (html, takeaways)"""
    if not any(any(v for v in r["phase"].values()) for r in rows):
        return [], section(14, "Over the episode: does the persona fade or grow?", "", "", "",
                       ["No per-turn data yet: rerun scripts/summarize_runs.py (it adds the readout per third of the "
                        "episode to the summaries)."], "")
    co_sel = lambda r: not r["steered"] and r["select"] in ("one", "count", "fraction") and r["condition"] != "none"
    cols = ["early", "mid", "late", "late − early", "co-players: early", "co-players: late", "co-players: late − early"]
    ph = lambda p: (lambda r, u: r["phase"][p][u])
    diff = lambda r, u: r["phase"]["late"][u] - r["phase"]["early"][u]
    get = {"early": (is_vector, ph("early")), "mid": (is_vector, ph("mid")), "late": (is_vector, ph("late")),
           "late − early": (is_vector, diff), "co-players: early": (co_sel, ph("early")),
           "co-players: late": (co_sel, ph("late")), "co-players: late − early": (co_sel, diff)}
    tab, found, q = matrix("Readout slope per third of the agent's turns (SD per unit c)", "trait", traits, cols,
                           lambda t, c: phase_slope(rows, t, get[c][0], get[c][1]))
    fades, grows, builds, ebbs = [], [], [], []
    for t in traits:
        e, d = found[(t, "early")], found[(t, "late − early")]
        # listed when significant and at least 10% of the early effect (a 3% change is significant at this n, not material)
        if e and d and sig(d, q[(t, "late − early")]) and math.isfinite(e["b"]) and e["b"] and abs(d["b"] / e["b"]) >= 0.1:
            share = d["b"] / e["b"]
            (grows if share > 0 else fades).append(f"{t} ({e['b']:+.2f} early → {found[(t, 'late')]['b']:+.2f} late, "
                                                  f"{100 * share:+.0f}%)")
        cd = found[(t, "co-players: late − early")]
        if cd and sig(cd, q[(t, "co-players: late − early")]):
            (builds if cd["b"] > 0 else ebbs).append(
                f"{t} ({found[(t, 'co-players: early')]['b']:+.2f} → {found[(t, 'co-players: late')]['b']:+.2f})")
    n = len({r["cluster"] for r in rows if any(_three(r, t) for t in traits)})
    f = [f"Episodes long enough for three thirds (≥ 3 monitored turns per agent): {n}."]
    f.append(("The persona <b>fades</b> over the episode (by ≥ 10% of its early effect) for: " + "; ".join(fades) + "."
              if fades else "No trait's effect fades materially (≥ 10%) over the episode.")
             + (" It <b>grows</b> for: " + "; ".join(grows) + "." if grows else ""))
    f.append(("Contagion builds up over the episode for: " + "; ".join(builds) + ". " if builds else "")
             + ("Contagion is strongest early and weakens over the episode for: " + "; ".join(ebbs)
                + " — the co-players react to the steered agent's first messages, then return toward their usual selves."
                if ebbs else "" if builds else "Contagion does not change significantly over the episode."))
    take = []
    if fades or grows:
        take.append("<b>The persona " + ("fades" if len(fades) >= len(grows) else "grows") + " over a game.</b> "
                    + (f"Pushed agents drift back toward their usual selves for {', '.join(x.split(' (')[0] for x in fades)}"
                       if fades else "") + ("; " if fades and grows else "")
                    + (f"the effect builds for {', '.join(x.split(' (')[0] for x in grows)}" if grows else "")
                    + " (section 14): one steering strength does not mean one persona strength throughout an episode.")
    if ebbs and not builds:
        take.append("<b>Contagion is a first reaction.</b> The co-players' drift toward the steered trait is largest in "
                    f"their first third of turns and weakens later ({', '.join(x.split(' (')[0] for x in ebbs)}; "
                    "section 14): they react to the steered agent's opening messages rather than converge on its persona.")
    if builds:
        take.append(f"<b>Contagion accumulates.</b> The co-players' drift toward {', '.join(x.split(' (')[0] for x in builds)} "
                    "grows from the first to the last third of their turns (section 14).")
    return take, section(14, "Over the episode: does the persona fade or grow?",
                   "Does the steered persona hold over a whole game, or does the conversation pull the agent back "
                   "(or push it further)? Does the co-players' drift toward the trait build up as they keep talking?",
                   "Each agent's turns are split into thirds (first, middle, last third of <i>its own</i> turns); the "
                   "readout is averaged per third, in SD units of the unsteered agents in the <i>same</i> third of the "
                   "same game (so a game's own drift, e.g. longer answers late, is removed). Per third: the slope over "
                   "the coefficient, as in section 2. late − early: the slope of each agent's change from its first to "
                   "its last third. Co-players: the unsteered agents of episodes where one agent or half are steered "
                   "(as in section 8). Only agents with at least 3 monitored turns.",
                   "late − early with the same sign as the early slope = the effect grows; opposite sign = it fades "
                   "(e.g. early +5, late − early −2: the agent drifts back toward its usual self). A rising co-players' "
                   "late − early = contagion accumulates over the conversation.", f, tab)


def actions_section(rows, traits, cells):
    """15: what the steered agents choose, per game: numeric choices and the share of each action type."""
    if not any(r["act_n"] for r in rows):
        return [], section(15, "Actions: what do steered agents do differently?", "", "", "",
                       ["No action data yet: rerun scripts/summarize_runs.py (it adds each agent's choices to the "
                        "summaries)."], "")
    out, hits, n_sig, n_all = [], [], 0, 0
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell and r["act_n"]]
        vec = [r for r in cr if is_vector(r) and r["team_cond"] is None]
        base = [r for r in cr if r["condition"] == "none"]
        if not vec or len(base) < 3:
            continue
        measures = []
        if sum(r["act_num"] is not None for r in base) >= len(base) / 2:
            # an agent mean far outside the game's range (a guess of 10^200) would swamp the slope: Tukey fence, k = 3
            q1, q3 = np.percentile([r["act_num"] for r in cr if r["act_num"] is not None], [25, 75])
            lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
            m0 = float(np.mean([r["act_num"] for r in base if r["act_num"] is not None and lo <= r["act_num"] <= hi]))
            measures.append((f"mean numeric choice (unsteered {m0:.3g})",
                             (lambda lo, hi: lambda r: r["act_num"] if r["act_num"] is not None and lo <= r["act_num"] <= hi
                              else None)(lo, hi), 1.0))
        total = defaultdict(float)
        for r in base:
            for lab, k in r["act_labels"].items():
                total[lab] += k
        n = sum(total.values()) or 1
        labs = [lab for lab, k in sorted(total.items(), key=lambda kv: -kv[1])
                if k / n >= 0.05 and lab not in ("text", "move", "number")][:6]
        if len(labs) >= 2 or (labs and total[labs[0]] / n < 0.95):
            for lab in labs:
                measures.append((f"share '{lab}' (unsteered {100 * total[lab] / n:.0f}%)",
                                 (lambda lab: lambda r: r["act_labels"].get(lab, 0) / r["act_n"])(lab), 100.0))
        if not measures:
            continue
        mrow = {m[0]: m for m in measures}

        def get(name, t):
            _, value, k = mrow[name]
            pts = [r for r in vec if r["trait"] == t and r["coef"] is not None] + base
            pts = [r for r in pts if value(r) is not None]
            s = slope([r["coef"] if r["condition"] != "none" else 0.0 for r in pts], [k * value(r) for r in pts],
                      [r["cluster"] for r in pts])
            return s

        tab, found, q = matrix(cell_name(cell), "action", list(mrow), traits, get)
        game_hits = sorted(((k, s) for k, s in found.items() if sig(s, q[k])), key=lambda kv: -abs(kv[1]["b"] / kv[1]["se"]))
        n_sig += len(game_hits)
        n_all += sum(1 for s in found.values() if s and math.isfinite(s.get("p", float("nan"))))
        hits += [f"{cell_name(cell)}: {k[1]} → {k[0].split(' (')[0]} {s['b']:+.2f}" for k, s in game_hits[:3]]
        out.append(tab)
    f = [f"{n_sig} of {n_all} game × action × trait slopes are significant."]
    if hits:
        f.append("Strongest per game: " + "; ".join(hits[:15]) + (f"; {len(hits) - 15} more" if len(hits) > 15 else "") + ".")
    take = [f"<b>Steering changes concrete choices.</b> {n_sig} of {n_all} game × action × trait slopes are "
            "significant (section 15); strongest: " + "; ".join(hits[:4]) + "."] if hits else []
    return take, section(15, "Actions: what do steered agents do differently?",
                   "Beyond the final reward: which choices change — bids, guesses and amounts, votes, play vs discard "
                   "vs hint, cooperate vs defect?",
                   "Each agent's parsed choices over the episode (default moves after an unparseable answer excluded). "
                   "<b>Numeric</b>: the mean of its numeric choices (a bid, a guess, an amount harvested; agents whose mean lies "
                   "beyond 3 interquartile ranges of the game's, e.g. a guess of 10^200, are left out). <b>Shares</b>: "
                   "the share of its choices of each type (an action label: APPROVE / REJECT, SUCCESS / FAIL, play / "
                   "discard / hint, TextArena's bracketed command with numbers replaced by #, e.g. 'bid #'). Only types "
                   "with ≥ 5% of the unsteered agents' choices. Steered agents with the persona vector (not the "
                   "evil-side settings) against the unsteered agents at c = 0: the slope per unit c.",
                   "Numeric rows are in the game's own units per unit c (e.g. +12 = bids 12 coins higher per unit c); "
                   "share rows in percentage points per unit c (+5 = that action 5 points more often per unit c, so +10 "
                   "at c = 2). Steered seats are drawn at random, so role-specific actions (only spies can FAIL a "
                   "mission) compare like with like on average. Labels are the game's own commands: single letters "
                   "e.g. market entry's [S]tay / [E]nter, GOPS's card ranks A / K / Q / J; '#' = a bare number "
                   "(a GOPS card value, a Mafia vote by number). Stars: Benjamini–Hochberg within each game's table.",
                   f, "".join(out))


def _shape(fit, q1, q2, cmax):
    """Shape of a reward curve from y = b1 c + b2 c²."""
    b1, b2 = fit[0], fit[1]
    s1, s2 = sig(b1, q1), sig(b2, q2)
    if s2 and b2["b"]:
        v = -b1["b"] / (2 * b2["b"])
        if abs(v) <= 0.25:  # turning point at no steering: either direction does the same
            return "∩ stronger either way hurts" if b2["b"] < 0 else "∪ stronger either way helps"
        if abs(v) < cmax:
            return ("∩ peaks" if b2["b"] < 0 else "∪ dips") + f" at c ≈ {v:+.1f}"
        return ("↑" if b1["b"] > 0 else "↓") + " saturating" if (b1["b"] > 0) != (b2["b"] > 0) else \
            ("↑" if b1["b"] > 0 else "↓") + " accelerating"
    if s1:
        return "↑ rising" if b1["b"] > 0 else "↓ falling"
    return ""


def _episode_mean(values, clusters):
    """(mean, SE, episodes) over per-episode means: each episode counts once, its agents averaged first."""
    by = defaultdict(list)
    for v, c in zip(values, clusters):
        if v is not None and math.isfinite(v):
            by[c].append(v)
    m = [float(np.mean(v)) for v in by.values()]
    if len(m) < 2:
        return (m[0] if m else float("nan")), float("nan"), len(m)
    return float(np.mean(m)), float(np.std(m, ddof=1)) / math.sqrt(len(m)), len(m)


def _misfit(pts, degree):
    """p that the means [(c, mean, SE)] scatter around their best polynomial of `degree` more than their SEs allow
    (weighted least squares, chi-square lack of fit)."""
    pts = [p for p in pts if math.isfinite(p[1]) and math.isfinite(p[2]) and p[2] > 0]
    df = len(pts) - degree - 1
    if df < 1 or len(pts) < 4:
        return float("nan")
    x, y, w = (np.array(v) for v in zip(*[(c, m, 1 / s ** 2) for c, m, s in pts]))
    X = np.vander(x, degree + 1)
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return chi2_sf(float((w * (y - X @ beta) ** 2).sum()), df)


def curve_checks(own_all, settings):
    """Per (game cell, trait, setting): the steered agents' reward change as a curve over c, described without a
    slope: the mean per c; whether it differs from 0 at all, from a straight line, from a smooth bend (beyond what
    the per-point SEs allow); and the mean for c < 0 and c > 0 (pushed down / up), mild (|c| <= 1) and strong.
    The side means are compared with 0 with SEs clustered by seed: every steered episode with the same seed is
    compared with the same unsteered episode, so a lucky or unlucky baseline moves them all (the per-c means share
    that offset, which cancels in the straight-line and bend checks)."""
    by = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    for r in own_all:
        if r["condition"] in settings and r["coef"] is not None:
            by[(r["cell"], r["trait"], r["condition"])][r["coef"]][0].append(r["diff"])
            by[(r["cell"], r["trait"], r["condition"])][r["coef"]][1].append(r["cluster"])
            by[(r["cell"], r["trait"], r["condition"])][r["coef"]][2].append(r["episode"])  # the seed
    out = {}
    for k, by_c in by.items():
        pts = {c: _episode_mean(*v[:2]) for c, v in sorted(by_c.items())}
        line = [(c, m, s) for c, (m, s, _) in pts.items()]
        res = {"pts": pts, "p_line": _misfit(line, 1), "p_bend": _misfit(line, 2)}
        for s in (-1, 1):
            pick = lambda keep: _episode_mean(*[sum((v[i] for c, v in by_c.items() if c * s > 0 and keep(abs(c))), [])
                                                for i in (0, 2)])
            m, se, n = pick(lambda a: True)
            res[s] = {"b": m, "se": se, "clusters": n, "n": n,
                      "p": math.erfc(abs(m / se) / math.sqrt(2)) if math.isfinite(se) and se > 0 else float("nan"),
                      "mild": pick(lambda a: a <= 1)[0], "strong": pick(lambda a: a > 1)[0]}
        out[k] = res
    keys = list(out)
    for name, q in (("q_line", bh([out[k]["p_line"] for k in keys])),
                    ("q_bend", bh([out[k]["p_bend"] for k in keys]))):
        for k, v in zip(keys, q):
            out[k][name] = v
    q = bh([out[k][s]["p"] for k in keys for s in (-1, 1)])
    for i, k in enumerate(keys):
        out[k][-1]["q"], out[k][1]["q"] = q[2 * i], q[2 * i + 1]
    return out


def side_pattern(ch):
    """Which way of pushing changes the reward: from the c < 0 and c > 0 means and their BH q."""
    dn = (1 if ch[-1]["b"] > 0 else -1) if sig(ch[-1], ch[-1]["q"]) else 0
    up = (1 if ch[1]["b"] > 0 else -1) if sig(ch[1], ch[1]["q"]) else 0
    if not dn and not up:
        return ""
    if dn and up:
        return "opposite ways (monotone)" if dn != up else "either push hurts" if dn < 0 else "either push helps"
    return ("only pushing down " if dn else "only pushing up ") + ("helps" if (dn or up) > 0 else "hurts")


SIDE_SETTINGS = SETTING_ORDER[:8]  # the vector settings: who carries the persona


def curve_id(cell, t):
    """Anchor of a game x trait row of section 16's individual curves."""
    return "c16-" + "-".join(cell) + "-" + t


OPEN_ON_LINK = ("<script>function openTarget(){var e=document.getElementById(decodeURIComponent(location.hash.slice(1)));"
                "if(!e)return;for(var d=e.closest('details');d;d=d.parentElement.closest('details'))d.open=true;"
                "e.scrollIntoView();}addEventListener('hashchange',openTarget);addEventListener('load',openTarget);"
                "</script>")


def _single_chart(title, pts, colour, span, note, caption):
    """One small chart of one curve: [(c, mean, half CI)] with c = 0 as the unsteered reference."""
    return (f"<div class=chart><h3>{html.escape(title)}</h3>"
            + line_chart({"curve": pts + [(0.0, 0.0, float("nan"))]}, title, "reward change", width=240, height=170,
                         colours={"curve": colour}, notes={"curve": note}, span=span)
            + f"<div class=muted style='font-size:11px'>{caption}</div></div>")


def setting_charts(checks, traits, cells, pooled, shapes) -> str:
    """Per game: the mean change when pushed down (c < 0) and up (c > 0), one dot per trait x setting; then per
    trait one row of separate charts: the curve pooled over the settings without a side (the one whose shape the
    table above gives) and one chart per setting, on a shared y axis."""
    colour = {c: f"var(--s{i + 1})" for i, c in enumerate(SIDE_SETTINGS)}
    ci = lambda pts: [(x, m, 1.96 * se) for x, (m, se, _) in pts.items()]
    out = ["<h3>Per game: every curve on its own</h3><p class=muted>Each game (click to open; a cell of the shape "
           "table above jumps here): first one chart per trait with every setting as a line (hover a line for its "
           "pushed-down and pushed-up means). Then one row per trait: "
           "<b>pooled</b> = all settings without a side (one, half, all, both_one) together, the curve whose shape "
           "the table gives; then one chart per setting (who is steered). Each: the steered agents' mean reward change "
           "at each c vs the same seats in the unsteered episode with the same seed, game units, 95% CIs, c = 0 the "
           "unsteered reference; the charts of a row share their y axis. Under each: down / up = the mean change over "
           "c &lt; 0 / c &gt; 0 (stars: differs from 0 after BH); 'bends' = the curve departs from a straight line "
           "beyond its noise (BH), otherwise its zigzag is noise.</p>", OPEN_ON_LINK]
    for cell in cells:
        here = [c for c in SIDE_SETTINGS if any((cell, t, c) in checks for t in traits)]
        if not here:
            continue
        legend = "".join(f'<span><i style="background:{colour[c]}"></i>{html.escape(c)}</span>' for c in here)
        rows_ = []
        for t in traits:
            sets = [c for c in here if (cell, t, c) in checks]
            if not sets:
                continue
            curves = [ci(checks[(cell, t, c)]["pts"]) for c in sets] + [ci(pooled.get((cell, t), {}))]
            ext = [v for pts in curves for x, m, h in pts if math.isfinite(m)
                   for v in (m - (h if math.isfinite(h) else 0), m + (h if math.isfinite(h) else 0))]
            span = (min(ext + [0]), max(ext + [0])) if ext else None
            charts = []
            if pooled.get((cell, t)):
                shape = shapes.get((cell, t)) or "flat"
                charts.append(_single_chart("pooled", ci(pooled[(cell, t)]), "var(--pos)", span,
                                            f"pooled over one, half, all, both_one: {shape}", f"shape: {html.escape(shape)}"))
            for c in sets:
                ch = checks[(cell, t, c)]
                bends = ch["q_line"] == ch["q_line"] and ch["q_line"] < 0.05
                summary = (f"down {ch[-1]['b']:+.2f}{stars(ch[-1]['q'])} / up {ch[1]['b']:+.2f}{stars(ch[1]['q'])}"
                           + (" · bends" if bends else ""))
                charts.append(_single_chart(c, ci(ch["pts"]), colour[c], span,
                                            f"{c}: {summary}; vs a straight line q {fmt(ch['q_line'], 3)}", summary))
            rows_.append(f"<h3 id='{curve_id(cell, t)}'>{html.escape(t)} <span class=muted>· "
                         f"{html.escape(shapes.get((cell, t)) or 'flat')} (pooled)</span></h3>"
                         f"<div class='charts mini'>{''.join(charts)}</div>")
        overview = []  # one chart per trait, every setting a line: the game at a glance
        for t in traits:
            sets = [c for c in here if (cell, t, c) in checks]
            if sets:
                series = {c: [(x, m, 1.96 * se) for x, (m, se, _) in checks[(cell, t, c)]["pts"].items()]
                          + [(0.0, 0.0, float("nan"))] for c in sets}
                note = {c: f"{c}: pushed down {checks[(cell, t, c)][-1]['b']:+.2f}, up {checks[(cell, t, c)][1]['b']:+.2f}"
                           + (" · bends" if checks[(cell, t, c)]["q_line"] < 0.05 else "") for c in sets}
                overview.append(f"<div class=chart><h3>{html.escape(t)}</h3>"
                                f"{line_chart(series, f'{cell_name(cell)} {t}', 'reward change', colours=colour, notes=note)}"
                                f"</div>")
        out.append(f"<details><summary>{html.escape(cell_name(cell))}: every curve</summary>"
                   f"<div class=legend>{legend}<span>bars: 95% CI</span></div>"
                   f"<h3>All settings per trait</h3><div class=charts>{''.join(overview)}</div>{''.join(rows_)}</details>")
    return "".join(out)


def _per_point(checks):
    """'~N episodes per point' (or a range) over the per-setting curves' points, for the texts that explain noise."""
    ns = sorted(n for ch in checks.values() for _, _, n in ch["pts"].values() if n)
    if not ns:
        return "few episodes per point"
    lo, hi = ns[len(ns) // 10], ns[-1 - len(ns) // 10]  # 10th-90th percentile: a stray short run does not set it
    return f"~{lo} episodes per point" if lo == hi else f"{lo}–{hi} episodes per point"


def _shape_reality(checks, cmax):
    """Findings and a takeaway for section 16: is the zigzag of the per-setting curves real, and which way of
    pushing matters."""
    if not checks:
        return [], []
    n = len(checks)
    q_ok = lambda k, name: checks[k][name] == checks[k][name] and checks[k][name] < 0.05
    anyeff = sum(1 for ch in checks.values() if side_pattern(ch))
    bent = [k for k in checks if q_ok(k, "q_line")]
    wiggly = [k for k in checks if q_ok(k, "q_bend")]
    tested = sum(1 for k in checks if checks[k]["p_line"] == checks[k]["p_line"])
    raw = sum(1 for k in checks if checks[k]["p_line"] < 0.05)
    pats = defaultdict(int)
    for ch in checks.values():
        pats[side_pattern(ch) or "neither side differs from 0"] += 1
    grow = []  # strong / mild on the sides that differ from 0
    for ch in checks.values():
        for s in (-1, 1):
            if sig(ch[s], ch[s]["q"]) and math.isfinite(ch[s]["mild"]) and ch[s]["mild"] and math.isfinite(ch[s]["strong"]):
                grow.append(ch[s]["strong"] / ch[s]["mild"])
    f = [f"<b>Is the bend real?</b> Of {n} curves (game × trait × setting), {anyeff} differ from 0 on at least one side; "
         f"{len(bent)} of {tested} depart from a straight line beyond what the per-point noise allows "
         f"({raw} at uncorrected p &lt; 0.05, ≈ {round(0.05 * tested)} expected by chance), {len(wiggly)} from a smooth "
         f"bend. The zigzag in most per-setting lines is sampling noise ({_per_point(checks)}), not a shape.",
         "<b>Which way of pushing matters</b> (mean change over all c &lt; 0 and over all c &gt; 0 vs 0, SEs clustered by "
         "seed, BH over all sides): "
         + "; ".join(f"{k}: {v}" for k, v in sorted(pats.items(), key=lambda kv: -kv[1])) + "."]
    if grow:
        f.append(f"<b>Mild vs strong:</b> where a side differs from 0, the effect at |c| &gt; 1 is {np.median(grow):.2f}× "
                 f"the effect at |c| ≤ 1 (median); it flips sign in {sum(1 for g in grow if g < 0)} of {len(grow)}: "
                 "stronger steering gives more of the same, not the opposite.")
    one_sided = sum(v for k, v in pats.items() if k.startswith("only"))
    take = [f"<b>Per setting, the zigzag is noise.</b> Only {len(bent)} of {tested} game × trait × setting reward curves "
            f"depart from a straight line beyond what {_per_point(checks)} allow; the bends in the pooled game × trait "
            "curves below need all settings together to show. Where a setting moves the reward, it is mostly "
            f"<b>one-sided</b>: {one_sided} curves change only when pushed one way, "
            f"{pats.get('opposite ways (monotone)', 0)} in opposite ways for the two directions, "
            f"{pats.get('either push hurts', 0) + pats.get('either push helps', 0)} the same way for both"
            + (f"; stronger steering gives more of the same ({np.median(grow):.1f}× the mild effect)" if grow else "")
            + " (section 16)."]
    return f, take


def _bend_table(checks):
    """The curves whose bend is real (not a straight line, after BH): means at each c."""
    bent = sorted((k for k in checks if checks[k]["q_line"] == checks[k]["q_line"] and checks[k]["q_line"] < 0.05),
                  key=lambda k: checks[k]["q_line"])
    if not bent:
        return "<h3>Curves that really bend</h3><p class=muted>none: every curve is consistent with a straight line</p>"
    coefs = sorted({c for k in bent for c in checks[k]["pts"]})
    rows_ = [f"<h3>Curves that really bend</h3><div class=wrap><table><tr><th class=l>game</th><th class=l>trait</th>"
             f"<th class=l>setting</th>" + "".join(f"<th>c = {c:g}</th>" for c in coefs) + "<th>q (vs line)</th></tr>"]
    for k in bent:
        pts = checks[k]["pts"]
        top = max([abs(m) for m, _, _ in pts.values() if math.isfinite(m)] or [1.0]) or 1.0
        rows_.append(f"<tr><td class=l>{html.escape(cell_name(k[0]))}</td><td class=l>{k[1]}</td><td class=l>{k[2]}</td>"
                     + "".join(_value_td(pts[c][0], 1.96 * pts[c][1], scale=top, d=2) if c in pts else "<td>–</td>" for c in coefs)
                     + f"<td>{fmt(checks[k]['q_line'], 4)}</td></tr>")
    rows_.append("</table></div><p class=muted>Mean reward change of the steered agents at each c (game units); "
                 "colour = sign, strength relative to the row's largest; hover for the 95% CI. The same curves as "
                 "charts:</p><div class='charts mini'>")
    colour = {c: f"var(--s{i + 1})" for i, c in enumerate(SIDE_SETTINGS)}
    for k in bent:
        pts = [(x, m, 1.96 * se) for x, (m, se, _) in checks[k]["pts"].items()]
        rows_.append(_single_chart(f"{cell_name(k[0])} · {k[1]} · {k[2]}", pts, colour.get(k[2], "var(--pos)"), None,
                                   f"q vs a straight line {fmt(checks[k]['q_line'], 4)}",
                                   f"<a href='#{curve_id(k[0], k[1])}'>all settings of this game × trait</a>"))
    rows_.append("</div>")
    return "".join(rows_)


def dose_response_section(steered, traits, cells, scale, own_all=()):
    """16: the steered agent's reward over the coefficient, per game and trait: where it saturates or flips."""
    coefs = sorted({r["coef"] for r in steered if r["coef"] is not None})
    cmax = max([abs(c) for c in coefs] or [2])
    charts = ["<div class=charts>"]
    for t in traits:
        tr = [r for r in steered if r["trait"] == t and r["coef"] is not None]
        pts = defaultdict(lambda: ([], []))
        for r in tr:
            pts[r["coef"]][0].append(r["diff"] / scale.get(r["cell"], 1.0))
            pts[r["coef"]][1].append(r["cluster"])
        curve = [(c, *mean_ci(*v)[:2]) for c, v in sorted(pts.items())] + [(0.0, 0.0, float("nan"))]
        charts.append(f"<div class=chart><h3>{html.escape(t)}</h3>"
                      f"{line_chart({'vector': curve}, t, 'reward change (SD)')}</div>")
    charts.append("</div>")
    fits = {}
    for cell in cells:
        for t in traits:
            cr = [r for r in steered if r["cell"] == cell and r["trait"] == t and r["coef"] is not None]
            if len({r["coef"] for r in cr}) >= 4:
                fits[(cell, t)] = ols([r["diff"] for r in cr], [[r["coef"], r["coef"] ** 2] for r in cr],
                                      [r["cluster"] for r in cr])
    keys = list(fits)
    q = bh([fits[k][0]["p"] for k in keys] + [fits[k][1]["p"] for k in keys])
    q1, q2 = dict(zip(keys, q[:len(keys)])), dict(zip(keys, q[len(keys):]))
    shapes = {k: _shape(fits[k], q1[k], q2[k], cmax) for k in keys}
    tab = ["<h3>Shape of the reward curve per game × trait</h3><div class=wrap><table><tr><th class=l>game</th>"
           + "".join(f"<th>{html.escape(t)}</th>" for t in traits) + "</tr>"]
    for cell in cells:
        if not any((cell, t) in shapes for t in traits):
            continue
        tds = []
        for t in traits:
            k = (cell, t)
            if k not in fits:
                tds.append("<td><span class=muted>–</span></td>")
                continue
            b1, b2 = fits[k]
            tip = (f"linear {b1['b']:+.3f} (q {q1[k]:.3f}), quadratic {b2['b']:+.3f} (q {q2[k]:.3f}), n {b1['n']}"
                   if math.isfinite(b1["b"]) else "not enough data")
            tds.append(f"<td title='{html.escape(tip)}'><a href='#{curve_id(cell, t)}'>"
                       f"{html.escape(shapes[k]) or '<span class=muted>flat</span>'}</a></td>")
        tab.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td>{''.join(tds)}</tr>")
    tab.append("</table></div>")
    per_game = ["<details><summary>Mean reward change at each coefficient, per game (game's own units)</summary>"]
    for cell in cells:
        cr = [r for r in steered if r["cell"] == cell and r["coef"] is not None]
        if not cr:
            continue
        rows_ = ["<h3>" + html.escape(cell_name(cell)) + "</h3><div class=wrap><table><tr><th class=l>trait</th>"
                 + "".join(f"<th>c = {c:g}</th>" for c in coefs) + "</tr>"]
        for t in traits:
            cells_ = []
            for c in coefs:
                v = [(r["diff"], r["cluster"]) for r in cr if r["trait"] == t and r["coef"] == c]
                cells_.append(mean_cell(mean_ci([a for a, _ in v], [b for _, b in v])) if v else "<td>–</td>")
            rows_.append(f"<tr><td class=l>{t}</td>{''.join(cells_)}</tr>")
        per_game.append("".join(rows_) + "</table></div>")
    per_game.append("</details>")
    peaks = [f"{cell_name(k[0])} × {k[1]} ({s})" for k, s in shapes.items() if s.startswith(("∩", "∪"))]
    sat = [f"{cell_name(k[0])} × {k[1]}" for k, s in shapes.items() if "saturating" in s]
    lin = sum(1 for s in shapes.values() if s.endswith(("rising", "falling")))
    f = [f"Of {len(shapes)} game × trait curves: {lin} change steadily with c, {len(sat)} saturate, {len(peaks)} peak or "
         f"dip inside the tested range, {sum(1 for s in shapes.values() if not s)} show no significant trend."]
    if peaks:
        f.append("Not monotonic (more steering is not more effect): " + "; ".join(peaks[:10])
                 + (f"; {len(peaks) - 10} more" if len(peaks) > 10 else "") + ".")
    if sat:
        f.append("Saturating: " + "; ".join(sat[:10]) + (f"; {len(sat) - 10} more" if len(sat) > 10 else "") + ".")
    hurt = [f"{cell_name(k[0])} × {k[1]}" for k, v in shapes.items() if v == "∩ stronger either way hurts"]
    checks = curve_checks(own_all, SIDE_SETTINGS)
    pooled = {}  # (cell, trait) -> {c: (mean, SE, episodes)}: the curve the shape table describes, game units
    for k, rs in _by([r for r in steered if r["coef"] is not None], lambda r: (r["cell"], r["trait"])).items():
        pooled[k] = {c: _episode_mean([r["diff"] for r in v], [r["cluster"] for r in v])
                     for c, v in sorted(_by(rs, lambda r: r["coef"]).items())}
    f_real, t_real = _shape_reality(checks, cmax)
    f += f_real
    take = t_real
    if peaks or sat:
        take.append(f"<b>More steering is not simply more effect.</b> Of {len(shapes)} game × trait reward curves, {lin} "
                    f"change steadily with c, {len(peaks)} bend inside ±{cmax:g}"
                    + (f" — in {len(hurt)} of them the agent does worse the harder it is pushed, in either direction "
                       f"(e.g. {'; '.join(hurt[:3])}): the model's default play is near its best there" if hurt else "")
                    + f"; {sum(1 for v in shapes.values() if not v)} show no trend (section 16).")
    return take, section(16, "Dose–response: how does the reward change with steering strength?",
                   "Does more steering give proportionally more effect, or does the effect level off, peak at a "
                   "moderate strength, or flip?",
                   "The steered agent's reward minus the same seat's in the unsteered episode with the same seed (as "
                   "in section 9), at each coefficient. Charts: all games pooled (SD units of each game's unsteered "
                   "rewards), with 95% CIs. Table: per game × trait, a fit y = b₁c + b₂c² (SEs clustered by episode, "
                   "Benjamini–Hochberg over the table): ↑/↓ rising/falling = only the straight part is significant; "
                   "saturating = it bends back but the turning point lies beyond ±" + f"{cmax:g}" + "; ∩/∪ = it peaks or "
                   "dips inside the range (turning point given); flat = no significant trend. Details: the mean reward "
                   "change at every coefficient, per game.",
                   "A ∩ means the trait helps up to a point and hurts beyond it — e.g. when moderate steering "
                   "changes the play but strong steering breaks the answers (section 13). '∩ stronger either way hurts': the turning point is at c ≈ 0, so "
                   "the harder the persona is pushed, in either direction, the worse the agent does. Every point is measured "
                   "against the same unsteered games, so a lucky or unlucky baseline shifts the whole curve up or down: the "
                   "fit has an intercept, so the <i>shape</i> (rising, bending) is robust to that, the level is not — with few unsteered episodes, trust shapes more "
                   "than levels. Hover a table cell for the fitted coefficients. Shapes need data "
                   f"at ≥ 4 coefficients. <b>Jagged lines are not a shape</b>: with {_per_point(checks)}, the means "
                   "zigzag by chance. 'Is the bend real?' tests each curve (game × trait × setting) against a straight "
                   "line and against a smooth bend, given each point's SE: only curves that fail the straight line "
                   "after correction really bend. 'Pushed down / up' summarises a curve without a slope: the mean "
                   "change over all c &lt; 0 and over all c &gt; 0.", f, "".join(charts) + "".join(tab) + _bend_table(checks) + setting_charts(checks, traits, cells, pooled, shapes)
                   + "".join(per_game))

# ---------------------------------------------------------------- 17-24: more analyses of the same rows

def _single_vector(r):
    """The steered agent of a one-agent persona-vector condition (no side, no prompt, no random control)."""
    return r["steered"] and r["via"] == "vector" and not r["random"] and r["select"] == "one" and r["team_cond"] is None


def _by(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def _rank_corr(x, y):
    """Spearman rank correlation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 4:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[keep])), np.argsort(np.argsort(y[keep]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _value_td(v, h=None, scale=4.0, d=1):
    """A table cell coloured by its value (red up, blue down); `h` = half CI shown in the tooltip."""
    if v is None or not math.isfinite(v):
        return "<td class=muted>–</td>"
    a = max(-1.0, min(1.0, v / scale))
    colour = "192,57,43" if a > 0 else "44,111,187"
    tip = f" title='± {h:.2f} (95% CI)'" if h is not None and math.isfinite(h) else ""
    return f'<td style="background:rgba({colour},{abs(a) * 0.55:.2f})"{tip}>{v:+.{d}f}</td>'


def footprint_section(rows, traits):
    """17: what each vector does to all monitored personas, not only its own trait."""
    sv = [r for r in rows if _single_vector(r)]
    if not sv:
        return [], section(17, "Footprint: what else does each vector change?", "", "", "", [], "")
    cmax = max(abs(r["coef"]) for r in sv)
    groups = _by([r for r in sv if abs(r["coef"]) == cmax], lambda r: (r["trait"], 1 if r["coef"] > 0 else -1))
    monitored = sorted({u for r in sv for u in r["mon"]})
    tab = [f"<div class=wrap><table><tr><th class=l>vector (trait, strength)</th><th>own trait's readout (SD)</th>"
           f"<th>own trait's rank among the {len(monitored)} personas</th><th class=l>personas moved most up (SD)</th>"
           "<th class=l>personas moved most down (SD)</th></tr>"]
    full = [f"<details><summary>All {len(monitored)} monitored personas (mean readout, SD units, at c = ±{cmax:g})</summary>"
            "<div class=wrap><table><tr><th class=l>persona</th>"
            + "".join(f"<th>{html.escape(t)} {'+' if s > 0 else '−'}</th>" for t in traits for s in (1, -1)) + "</tr>"]
    fps, not_top = {}, []
    for t in traits:
        for s in (1, -1):
            fp = fingerprint(groups.get((t, s), []))
            fps[(t, s)] = fp
            if t not in fp:
                continue
            order = sorted(fp, key=lambda u: -s * fp[u])
            rank = order.index(t) + 1
            if rank > 1:
                not_top.append(f"{t} {'+' if s > 0 else '−'} (own trait ranks {rank}; {order[0]} moves most, {fp[order[0]]:+.1f})")
            up = sorted(fp, key=lambda u: -fp[u])[:4]
            down = sorted(fp, key=lambda u: fp[u])[:3]
            tab.append(f"<tr><td class=l>{t} c = {s * cmax:+g}</td>{_value_td(fp[t])}<td>{rank}</td>"
                       f"<td class=l>{', '.join(f'{u} {fp[u]:+.1f}' for u in up)}</td>"
                       f"<td class=l>{', '.join(f'{u} {fp[u]:+.1f}' for u in down)}</td></tr>")
    g17 = games_note({r["cell"] for g in groups.values() for r in g})
    tab.append("</table></div>" + g17)
    for u in monitored:
        full.append(f"<tr><td class=l>{html.escape(u)}</td>"
                    + "".join(_value_td(fps[(t, s)].get(u)) for t in traits for s in (1, -1)) + "</tr>")
    full.append("</table></div>" + g17 + "</details>")
    n_cases = sum(1 for k in fps if k[0] in fps[k])
    f = [f"The steered trait is the most-moved of all {len(monitored)} monitored personas in "
         f"{n_cases - len(not_top)} of {n_cases} vector × sign cases."]
    if not_top:
        f.append("Where another persona moves more than the steered one: " + "; ".join(not_top) + ".")
    take = [f"<b>Each vector moves a whole persona, not one trait.</b> Over all {len(monitored)} monitored personas, the "
            f"steered trait moves most in only {n_cases - len(not_top)} of {n_cases} cases (section 17)"
            + (f"; e.g. {not_top[0]}" if not_top else "") + ": name a vector by its footprint, not its label."]
    return take, section(17, "Footprint: what else does each vector change?",
                         "The report so far reads 8 traits; the monitor reads all monitored personas. What does each "
                         "persona vector do to the others — does 'evil' read as scheming, or as rude?",
                         f"Steered agents of the one-agent persona-vector condition at the strongest coefficient (c = "
                         f"±{cmax:g}): mean readout of every monitored persona, in SD units of the unsteered agents of "
                         "the same game (as in section 2). The rank: where the steered trait itself comes among all "
                         "personas, by movement in the steered direction.",
                         "Rank 1 = the vector's strongest effect is its own trait. A high rank means the label "
                         "undersells what the vector does: read 'moved most' as its actual content. The collapsible "
                         "table has every persona × vector.", f, "".join(tab) + "".join(full))


def naming_section(rows, traits):
    """18: can the monitor name the steered trait (not only flag the agent)?"""
    sv = [r for r in rows if _single_vector(r) and r["mon"]]
    if len(sv) < 100:
        return [], section(18, "Naming the trait: can a monitor tell which persona was injected?", "", "", "", [], "")
    monitored = sorted({u for r in sv for u in r["mon"]})

    def classify(us):
        X = np.nan_to_num(np.array([[r["mon"].get(u, 0.0) for u in us] for r in sv]))
        classes = sorted({(r["trait"], r["coef"] > 0) for r in sv})
        y = np.array([classes.index((r["trait"], r["coef"] > 0)) for r in sv])
        fold = np.array([r["episode"] % 2 for r in sv])  # held-out deals: episodes share seeds across conditions
        pred = np.zeros(len(sv), int)
        for k in (0, 1):
            tr, te = fold != k, fold == k
            C = np.stack([X[tr & (y == j)].mean(0) if (tr & (y == j)).any() else np.full(len(us), 1e6)
                          for j in range(len(classes))])
            pred[te] = (-2 * X[te] @ C.T + (C * C).sum(1)).argmin(1)
        return np.array([classes[j][0] for j in pred])

    truth = np.array([r["trait"] for r in sv])
    pred = classify(monitored)
    pred8 = classify(traits)
    absc = np.array([abs(r["coef"]) for r in sv])
    acc = lambda m: float((pred[m] == truth[m]).mean()) if m.any() else float("nan")
    chance = 1 / len(traits)
    by_c = {c: acc(absc == c) for c in sorted(set(absc))}
    tab = ["<h3>Accuracy by steering strength</h3><div class=wrap><table><tr><th class=l>|c|</th>"
           + "".join(f"<th>{c:g}</th>" for c in by_c) + "<th>all</th></tr><tr><td class=l>trait named correctly</td>"
           + "".join(f"<td>{100 * v:.0f}%</td>" for v in by_c.values()) + f"<td>{100 * acc(absc > 0):.0f}%</td></tr></table></div>"
           + games_note({r["cell"] for r in sv})]
    tab.append("<h3>Confusion: true trait (rows) → named trait (columns), % of the row</h3><div class=wrap><table><tr>"
               "<th class=l>true \\ named</th>" + "".join(f"<th>{t}</th>" for t in traits) + "</tr>")
    recall = {}
    for t in traits:
        m = truth == t
        shares = {u: float((pred[m] == u).mean()) if m.any() else float("nan") for u in traits}
        recall[t] = shares[t]
        tab.append(f"<tr><td class=l>{t}</td>" + "".join(
            f"<td style='background:rgba(192,57,43,{0.6 * v:.2f})'>{100 * v:.0f}%</td>" if math.isfinite(v)
            else "<td>–</td>" for v in shares.values()) + "</tr>")
    tab.append("</table></div>" + games_note({r["cell"] for r in sv}))
    sink = max(traits, key=lambda u: float((pred == u).mean()))
    f = [f"Named correctly: {100 * acc(absc > 0):.0f}% of steered agents (chance {100 * chance:.0f}%); "
         + ", ".join(f"{100 * v:.0f}% at |c| = {c:g}" for c, v in by_c.items()) + ".",
         "Share of each trait's agents named right: " + ", ".join(f"{t} {100 * v:.0f}%" for t, v in
                                                                  sorted(recall.items(), key=lambda kv: -kv[1])) + ".",
         f"Most errors go to <b>{sink}</b> ({100 * float((pred == sink).mean()):.0f}% of all guesses): the shared persona "
         "axis of section 3 makes it the default guess.",
         f"With only the {len(traits)} steered traits' readouts instead of all {len(monitored)}: "
         f"{100 * float((pred8 == truth).mean()):.0f}% — the other personas add little."]
    take = [f"<b>Naming the persona is possible but unreliable.</b> From its readout alone, the injected trait is named "
            f"right for {100 * acc(absc > 0):.0f}% of steered agents (chance {100 * chance:.0f}%; "
            f"{100 * by_c[min(by_c)]:.0f}% at |c| = {min(by_c):g}, {100 * by_c[max(by_c)]:.0f}% at |c| = "
            f"{max(by_c):g}: strong personas are far easier to name); {min(recall, key=recall.get)} is hardest "
            f"({100 * min(recall.values()):.0f}%) and most mistakes are named {sink} (section 18)."]
    return take, section(18, "Naming the trait: can a monitor tell which persona was injected?",
                         "Section 4 shows a monitor can flag the steered agent. Can it also say <i>which</i> persona "
                         "was injected, and at what strength does that start to work?",
                         f"Steered agents of the one-agent persona-vector condition. Each is described by its readout of "
                         f"all {len(monitored)} monitored personas (SD units). A nearest-centroid classifier learns the "
                         f"average readout of each trait × sign (16 classes: the monitor does not know whether the trait "
                         "was pushed up or down) on even-numbered episodes and names the trait of agents in "
                         "odd-numbered ones, and vice versa — the test deals were never seen in training.",
                         "Accuracy above chance = the readout carries the persona's identity. The confusion table shows "
                         "which traits are mistaken for which; a column that collects many errors is a 'default' "
                         "persona the readout drifts to whatever is steered.", f, "".join(tab))


ARMS = {"vector": "var(--pos)", "prompt": "var(--neg)", "both": "var(--s3)"}  # how the persona is given
ARM_OF = {"one": "vector", "prompt_one": "prompt", "both_one": "both", "all": "vector", "prompt_all": "prompt",
          "evil_team": "vector", "evil_team_prompt": "prompt"}


def _arm_point(rs, value, s, c=None):
    """(mean, SE, episodes) of value(r) over the rows on side s (sign of c), at |c| = c if given."""
    vals = [(value(r), r["cluster"]) for r in rs if (r["coef"] or 0) * s > 0 and (c is None or abs(r["coef"]) == c)]
    vals = [(v, cl) for v, cl in vals if v is not None and math.isfinite(v)]
    return _episode_mean([v for v, _ in vals], [cl for _, cl in vals])


def prompt_outcomes(rows, base, scale):
    """j (section 19): does the prompt get better outcomes than the vector? Per game × trait × side, the paired
    change of one steered agent's own reward (prompt_one vs one) and of the group's reward when everyone is steered
    (prompt_all vs all; games without hidden roles), SD units, against the vector at c = ±1 (matched) and at its best
    |c| (more chances: an optimistic reference). (findings, html)"""
    roles = role_cells(rows)
    one = paired(rows, base, lambda r: r["steered"] and r["condition"] in ("one", "prompt_one") and not r["random"])
    one = [{**r, "chg": r["diff"] / scale.get(r["cell"], 1.0)} for r in one]
    grp = welfare([r for r in rows if r["condition"] in ("all", "prompt_all") and not r["random"]
                   and r["team_cond"] is None and r["cell"] not in roles],
                  [r for r in rows if r["condition"] == "none" and r["cell"] not in roles])
    grp = [{**r, "chg": r["diff"]} for r in grp]
    findings, parts = [], []
    for label, data, vec, pr in (("one agent steered: its own result", one, "one", "prompt_one"),
                                 ("everyone steered: the group's result (games without hidden roles)", grp, "all",
                                  "prompt_all")):
        cell_ = {}  # (cell, trait, side, arm, |c|) -> (mean, SE, n, p)
        for k, rs in _by([r for r in data if r["coef"]], lambda r: (r["cell"], r["trait"], 1 if r["coef"] > 0 else -1,
                                                                    r["condition"], abs(r["coef"]))).items():
            m, se, n = _episode_mean([r["chg"] for r in rs], [r["cluster"] for r in rs])
            if n >= 5 and math.isfinite(se):
                cell_[k] = (m, se, n, math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else (0.0 if m else 1.0))
        q = dict(zip(cell_, bh([v[3] for v in cell_.values()])))
        keys = sorted({k[:3] for k in cell_ if k[3] == pr and (k[:3] + (vec, 1.0)) in cell_})
        arms = {"prompt": lambda k: k + (pr, 1.0), f"vector, c = ±1": lambda k: k + (vec, 1.0)}
        best = {k: max((x for x in cell_ if x[:4] == k + (vec,)), key=lambda x: cell_[x][0]) for k in keys}
        arms["vector, best |c| per cell"] = lambda k: best[k]
        tab = [f"<h4>{html.escape(label)}</h4><div class=wrap><table><tr><th class=l>how the persona is given</th>"
               "<th>game × trait × side cells</th><th>significant gains (BH)</th><th>significant losses (BH)</th>"
               "<th>mean change (SD)</th><th>share of cells with a gain</th><th>best gain in a game (SD), median over "
               "games</th></tr>"]
        summary = {}
        for a, kf in arms.items():
            v = [(k, cell_[kf(k)], q[kf(k)]) for k in keys]
            gains = [(k, x) for k, x, qq in v if qq < 0.05 and x[0] > 0]
            losses = [(k, x) for k, x, qq in v if qq < 0.05 and x[0] < 0]
            per_game = defaultdict(list)
            for k, x, _ in v:
                per_game[k[0]].append(x[0])
            top = float(np.median([max(xs) for xs in per_game.values()])) if per_game else float("nan")
            summary[a] = (gains, losses, per_game)
            tab.append(f"<tr><td class=l>{html.escape(a)}</td><td>{len(v)}</td><td>{len(gains)}</td><td>{len(losses)}</td>"
                       f"<td>{np.mean([x[0] for _, x, _ in v]):+.2f}</td><td>{100 * np.mean([x[0] > 0 for _, x, _ in v]):.0f}%"
                       f"</td><td>{top:+.2f}</td></tr>")
        tab.append("</table></div>")
        # head to head in the same cell: prompt minus the vector at c = ±1
        hh = {}
        for k in keys:
            (mp, sp, _, _), (mv, sv, _, _) = cell_[arms["prompt"](k)], cell_[arms["vector, c = ±1"](k)]
            se = math.hypot(sp, sv)
            hh[k] = (mp - mv, math.erfc(abs((mp - mv) / se) / math.sqrt(2)) if se > 0 else 1.0)
        hq = dict(zip(hh, bh([v[1] for v in hh.values()])))
        better = [k for k in hh if hq[k] < 0.05 and hh[k][0] > 0]
        worse = [k for k in hh if hq[k] < 0.05 and hh[k][0] < 0]
        wins = sum(1 for g in summary["prompt"][2] if max(summary["prompt"][2][g]) > max(summary["vector, c = ±1"][2][g]))
        name = lambda k: f"{cell_name(k[0])} × {k[1]} {'+' if k[2] > 0 else '−'}"
        gl = lambda a: "; ".join(f"{name(k)} {x[0]:+.2f} (n = {x[2]})" for k, x in
                                 sorted(summary[a][0], key=lambda kx: -kx[1][0])[:6]) or "none"
        tab.append(f"<p class=muted>Significant gains — prompt: {html.escape(gl('prompt'))}. Vector at c = ±1: "
                   f"{html.escape(gl('vector, c = ±1'))}.</p>" + games_note({k[0] for k in keys}))
        parts.append("".join(tab))
        n_g = len(summary["prompt"][2])
        findings.append(
            f"<b>Outcomes, {html.escape(label)}:</b> the prompt gives {len(summary['prompt'][0])} significant gains and "
            f"{len(summary['prompt'][1])} losses over {len(keys)} game × trait × side cells, the vector at c = ±1 "
            f"{len(summary['vector, c = ±1'][0])} and {len(summary['vector, c = ±1'][1])}; head to head in the same cell "
            f"the prompt is better in {100 * np.mean([v[0] > 0 for v in hh.values()]) if hh else float('nan'):.0f}% "
            f"of cells, significantly better in {len(better)} and worse in {len(worse)}; its best gain in a game beats "
            f"the matched vector's in {wins} of {n_g} games. With its strength chosen per cell the vector reaches "
            f"{len(summary['vector, best |c| per cell'][0])} significant gains (an optimistic reference: the best of "
            "several noisy strengths).")
    html_ = ("<h3>j. Outcomes: does the prompt get better results than the vector?</h3><p class=muted>Per game × trait "
             "× side: the paired change against the same seats (one agent) or the same episode (everyone) unsteered "
             "with the same seed, in SD units of the game; mean over episodes; Benjamini–Hochberg over every cell of "
             "the comparison. 'Best |c| per cell' picks, for each cell, the vector strength that did best, so it is "
             "biased upwards; it shows how much room a tuned strength has, not a fair contest.</p>" + "".join(parts))
    return findings, html_


def prompt_vector_section(rows, traits, base, scale):
    """19: the persona as a vector, as a prompt, or both, side by side at matched strength (c = ±1)."""
    one = [r for r in rows if r["steered"] and r["condition"] in ("one", "prompt_one", "both_one") and not r["random"]]
    if not any(r["condition"] == "prompt_one" for r in one):
        return [], section(19, "Vector, prompt or both?", "", "", "", [], "")
    arm = _by(one, lambda r: (ARM_OF[r["condition"]], r["trait"]))
    rew = _by(paired(rows, base, lambda r: r["steered"] and r["condition"] in ARM_OF and not r["random"]),
              lambda r: (r["condition"], r["trait"]))
    co = _by([r for r in rows if not r["steered"] and r["condition"] in ("one", "prompt_one", "both_one")],
             lambda r: (ARM_OF[r["condition"]], r["trait"]))
    none = [r for r in rows if r["condition"] == "none"]
    len0 = {c: float(np.mean([math.log(max(r["chars"], 1)) for r in rs if r["chars"] is not None] or [float("nan")]))
            for c, rs in _by(none, lambda r: r["cell"]).items()}
    fb0 = 100 * sum(r["fallbacks"] for r in none) / max(1, sum(r["turns"] for r in none))
    readout = lambda t: lambda r: r["mon"].get(t)
    reward = lambda r: r["diff"] / scale.get(r["cell"], 1.0)
    length = lambda r: (100 * (math.exp(math.log(max(r["chars"], 1)) - len0[r["cell"]]) - 1)
                        if r["chars"] is not None and math.isfinite(len0.get(r["cell"], float("nan"))) else None)
    unparse = lambda r: 100 * r["fallbacks"] / r["turns"] if r["turns"] else None
    cond_of = {"vector": "one", "prompt": "prompt_one", "both": "both_one"}
    at = lambda a: None if a == "prompt" else 1.0  # the prompt has no strength: compared with |c| = 1
    g19 = games_note({r["cell"] for r in one}, "Games")

    # a. over the dose: readout and reward for vector and both, the prompt as squares at ±1
    charts = {"readout": [], "reward": []}
    for t in traits:
        for kind, rs_of, val, ylab in (("readout", lambda a: arm.get((a, t), []), readout(t), f"{t} readout (SD)"),
                                       ("reward", lambda a: rew.get((cond_of[a], t), []), reward, "reward change (SD)")):
            series = {}
            for a in ARMS:
                rs = rs_of(a)
                cs = sorted({r["coef"] for r in rs if r["coef"]})
                if not cs:
                    continue
                pts = [(c, m, 1.96 * se) for c in cs for m, se, _ in [_arm_point(rs, val, 1 if c > 0 else -1, abs(c))]]
                series[a] = pts + ([] if a == "prompt" else [(0.0, 0.0, float("nan"))])
            if series:
                charts[kind].append(f"<div class=chart><h3>{html.escape(t)}</h3>"
                                    f"{line_chart(series, f'{t} {kind}', ylab, colours=ARMS)}</div>")

    # b-f. matched comparison per trait x side
    cols = (("readout of the trait (SD)", readout, lambda a, t: arm.get((a, t), []), 10.0),
            ("reward change (SD)", lambda t: reward, lambda a, t: rew.get((cond_of[a], t), []), 0.5),
            ("unparseable answers (% of turns)", lambda t: unparse, lambda a, t: arm.get((a, t), []), 20.0),
            ("answer length vs unsteered (%)", lambda t: length, lambda a, t: arm.get((a, t), []), 200.0),
            ("co-players' readout of the trait (SD)", readout, lambda a, t: co.get((a, t), []), 0.5))
    res = {}
    for t in traits:
        for s in (1, -1):
            for i, (_, val, rs_of, _) in enumerate(cols):
                for a in ARMS:
                    res[(t, s, i, a)] = _arm_point(rs_of(a, t), val(t), s, at(a))
            res[(t, s, 0, "vector2")] = _arm_point(arm.get(("vector", t), []), readout(t), s, 2.0)
    tabs = []
    for i, (title, _, _, sc) in enumerate(cols):
        extra = "<th>vector, c = ±2</th><th>both ÷ (vector + prompt)</th>" if i == 0 else ""
        tab = [f"<h3>{chr(98 + i)}. {html.escape(title)}</h3><div class=wrap><table><tr><th class=l>trait</th>"
               "<th>side (+ = toward the trait)</th><th>vector alone (c = ±1)</th><th>prompt alone</th>"
               "<th>prompt + vector (c = ±1)</th>" + extra + "</tr>"]
        for t in traits:
            for s in (1, -1):
                v = [res[(t, s, i, a)] for a in ARMS]
                tds = "".join(_value_td(m, 1.96 * se, scale=sc, d=1 if sc >= 10 else 2) for m, se, _ in v)
                if i == 0:
                    tot = v[0][0] + v[1][0]
                    ratio = v[2][0] / tot if math.isfinite(tot) and s * tot > 0.5 else float("nan")
                    w = res[(t, s, 0, "vector2")]
                    tds += _value_td(w[0], 1.96 * w[1], scale=sc) + (
                        f"<td>{ratio:.2f}</td>" if math.isfinite(ratio) else "<td class=muted>–</td>")
                tab.append(f"<tr><td class=l>{t}</td><td>{'+' if s > 0 else '−'}</td>{tds}</tr>")
        tab.append("</table></div>" + g19)
        tabs.append("".join(tab))

    # g. per game: how far each arm moves the trait the intended way (sign-aligned, mean over traits x sides)
    by_game = defaultdict(lambda: defaultdict(list))
    for (a, t), rs in arm.items():
        for r in rs:
            if t in r["mon"] and math.isfinite(r["mon"][t]) and (at(a) is None or abs(r["coef"]) == at(a)):
                by_game[r["group"]][a].append(math.copysign(1, r["coef"]) * r["mon"][t])
    games = sorted(g for g in by_game if all(by_game[g][a] for a in ARMS))
    gm = {g: {a: float(np.mean(by_game[g][a])) for a in ARMS} for g in games}
    p_beats = [g for g in games if gm[g]["prompt"] > gm[g]["vector"]]
    b_beats = [g for g in games if gm[g]["both"] > max(gm[g]["prompt"], gm[g]["vector"])]
    gtab = ["<h3>g. Per game: how far each moves the trait the intended way</h3><div class=wrap><table><tr>"
            "<th class=l>game</th><th>vector (c = ±1)</th><th>prompt</th><th>both (c = ±1)</th></tr>"]
    for g in games:
        gtab.append(f"<tr><td class=l>{html.escape(g)}</td>"
                    + "".join(_value_td(gm[g][a], None, scale=20) for a in ARMS) + "</tr>")
    gtab.append("</table></div><p class=muted>Readout of the steered trait × the sign of the push (SD vs unsteered), "
                "mean over traits and both sides: higher = moved further the intended way.</p>")

    # i. synergy: is prompt + vector more than the sum of the two? interaction = both - vector - prompt (all three
    # against the same unsteered reference), per trait x side, for the readout and the reward; BH over all of them
    syn = {}
    for t in traits:
        for sd in (1, -1):
            for i in (0, 1):
                (mb, sb, nb), (mv, sv, nv), (mp, sp, np_) = (res[(t, sd, i, a)] for a in ("both", "vector", "prompt"))
                d, se = mb - mv - mp, math.sqrt(sb ** 2 + sv ** 2 + sp ** 2)
                syn[(t, sd, i)] = {"b": d, "se": se, "n": nb + nv + np_,
                                   "p": math.erfc(abs(d / se) / math.sqrt(2)) if math.isfinite(se) and se > 0
                                   else float("nan")}
    syn_q = dict(zip(syn, bh([v["p"] for v in syn.values()])))
    stab = ["<h3>i. Synergy: is prompt + vector more than the sum of the two?</h3><div class=wrap><table><tr>"
            "<th class=l>trait</th><th>side</th><th>readout: both − (vector + prompt) (SD)</th>"
            "<th>reward: both − (vector + prompt) (SD)</th></tr>"]
    for t in traits:
        for sd in (1, -1):
            stab.append(f"<tr><td class=l>{t}</td><td>{'+' if sd > 0 else '−'}</td>"
                        + "".join(slope_cell({**syn[(t, sd, i)], "b": sd * syn[(t, sd, i)]["b"]} if i == 0 else
                                             syn[(t, sd, i)], syn_q[(t, sd, i)]) for i in (0, 1)) + "</tr>")
    stab.append("</table></div><p class=muted>Readout column sign-aligned: positive = together they push the trait "
                "further the intended way than the two separately would (synergy), negative = less (they overlap or "
                "compete), 0 = they simply add. Reward column: the extra reward change of the combination beyond the "
                "sum (SD units). Stars: Benjamini–Hochberg over the table; hover for SE and q.</p>" + g19)
    syn_game = {g: gm[g]["both"] - gm[g]["vector"] - gm[g]["prompt"] for g in games}

    # h. groups: everyone steered and the whole deceiving side, vector (c = ±1) vs prompt
    grp = _by([r for r in rows if r["steered"] and not r["random"]
               and r["condition"] in ("all", "prompt_all", "evil_team", "evil_team_prompt")],
              lambda r: (r["condition"], r["trait"]))
    ttab = ["<h3>h. Everyone steered, and the whole deceiving side: readout of the trait (SD)</h3><div class=wrap>"
            "<table><tr><th class=l>trait</th><th>side</th><th>all: vector (c = ±1)</th><th>all: prompt</th>"
            "<th>deceiving side: vector (c = ±1)</th><th>deceiving side: prompt</th></tr>"]
    g_more = g_n = 0
    for t in traits:
        for s in (1, -1):
            v = [_arm_point(grp.get((c, t), []), readout(t), s, at(ARM_OF[c]))
                 for c in ("all", "prompt_all", "evil_team", "evil_team_prompt")]
            for vec_, pr_ in ((v[0], v[1]), (v[2], v[3])):
                if math.isfinite(vec_[0]) and math.isfinite(pr_[0]):
                    g_n += 1
                    g_more += s * pr_[0] > s * vec_[0]
            ttab.append(f"<tr><td class=l>{t}</td><td>{'+' if s > 0 else '−'}</td>"
                        + "".join(_value_td(m, 1.96 * se, scale=10) for m, se, _ in v) + "</tr>")
    ttab.append("</table></div>" + games_note({r["cell"] for rs in grp.values() for r in rs}))

    # findings
    sides = [(t, s) for t in traits for s in (1, -1)]
    al = lambda t, s, i, a: s * res[(t, s, i, a)][0]
    stronger = [(t, s) for t, s in sides if al(t, s, 0, "prompt") > al(t, s, 0, "vector")]
    as_c2 = [(t, s) for t, s in sides if al(t, s, 0, "prompt") >= al(t, s, 0, "vector2")]
    wrong = [f"{t} {'+' if s > 0 else '−'} ({res[(t, s, 0, 'prompt')][0]:+.1f})" for t, s in sides
             if res[(t, s, 0, "prompt")][2] and al(t, s, 0, "prompt") <= 0]
    both_top = [(t, s) for t, s in sides if al(t, s, 0, "both") > max(al(t, s, 0, "prompt"), al(t, s, 0, "vector"))]
    f = [f"<b>Strength at matched dose:</b> the prompt moves the trait's readout further than the vector at c = ±1 for "
         f"{len(stronger)} of {len(sides)} trait × side cases, and at least as far as the vector at c = ±2 for "
         f"{len(as_c2)}; per game, further than the vector in {len(p_beats)} of {len(games)} games (table g)."]
    if wrong:
        f.append(f"<b>Prompts that fail:</b> {', '.join(wrong)}: the instruction does not move the readout the intended "
                 "way ('do not be X' names X); the vector does.")
    f.append(f"<b>Both together</b> move the readout furthest in {len(both_top)} of {len(sides)} cases and in "
             f"{len(b_beats)} of {len(games)} games: they stack (ratio column: ≈ 1 additive, &gt; 1 reinforcing).")
    kind = lambda i, sign: [f"{t} {'+' if sd > 0 else '−'} ({sd * syn[(t, sd, i)]['b']:+.1f})" if i == 0 else
                            f"{t} {'+' if sd > 0 else '−'} ({syn[(t, sd, i)]['b']:+.2f})" for t in traits for sd in (1, -1)
                            if sig(syn[(t, sd, i)], syn_q[(t, sd, i)])
                            and sign * (sd if i == 0 else 1) * syn[(t, sd, i)]["b"] > 0]
    syn_r, sub_r, syn_w, sub_w = kind(0, 1), kind(0, -1), kind(1, 1), kind(1, -1)
    f.append(f"<b>Synergy (table i):</b> on the readout, prompt + vector is significantly <i>more</i> than the sum of the "
             f"two for {len(syn_r)} of {len(sides)} trait × side cases{' (' + ', '.join(syn_r) + ')' if syn_r else ''} "
             f"and <i>less</i> for {len(sub_r)}{' (' + ', '.join(sub_r) + ')' if sub_r else ''}; the rest add up. Per "
             f"game (mean over traits and sides), more than the sum in {sum(1 for v in syn_game.values() if v > 0)} of "
             f"{len(syn_game)} games. On the reward, the combination differs from the sum in "
             f"{len(syn_w) + len(sub_w)} cases"
             + (f" (more: {', '.join(syn_w) or '–'}; less: {', '.join(sub_w) or '–'})" if syn_w or sub_w else "")
             + ". A readout is a projection that can saturate, so 'less than the sum' on it can be a ceiling, not "
               "competition. These verdicts pool all games; per game, and whether the games agree: section 25e.")
    ri = lambda a: [res[(t, s, 1, a)][0] for t, s in sides]
    r_pv, r_bv, r_bp = pearson(ri("prompt"), ri("vector")), pearson(ri("both"), ri("vector")), pearson(ri("both"), ri("prompt"))
    rg = defaultdict(list)  # per game x trait x side: the arms' reward effects, to see whether they agree
    for a in ARMS:
        for t in traits:
            for r in rew.get((cond_of[a], t), []):
                if at(a) is None or abs(r["coef"]) == at(a):
                    rg[(r["cell"], t, r["coef"] > 0, a)].append(reward(r))
        for r in rew.get(("one", t), []):  # the noise ceiling: the vector at c = ±2 against itself at c = ±1
            if abs(r["coef"]) == 2.0:
                rg[(r["cell"], t, r["coef"] > 0, "vector2")].append(reward(r))
    keys = sorted({k[:3] for k in rg})
    per = {a: [float(np.mean(rg[k + (a,)])) if rg.get(k + (a,)) else float("nan") for k in keys]
           for a in tuple(ARMS) + ("vector2",)}
    gp, gv = pearson(per["prompt"], per["vector"]), pearson(per["vector2"], per["vector"])
    gp2 = pearson(per["prompt"], per["vector2"])  # the prompt reads like a strong vector: compare it with c = ±2 too
    same_as_noise = math.isfinite(gp[0]) and math.isfinite(gv[0]) and gp[0] >= gv[0] - 0.1
    like_strong = math.isfinite(gp2[0]) and math.isfinite(gv[0]) and gp2[0] >= gv[0] - 0.1
    f.append(f"<b>Play:</b> per game × trait × side, the prompt's reward effect correlates r = {fmt(gp[0], 2)} with the "
             f"vector's (n = {gp[3]}); the vector at c = ±1 and at ±2 correlate r = {fmt(gv[0], 2)}, the most two "
             "near-identical interventions can agree given the noise"
             + (": prompt and vector agree about as well as the vector agrees with itself. " if same_as_noise else
                f". Against the vector at c = ±2 (which its readout effect resembles more) the prompt correlates "
                f"r = {fmt(gp2[0], 2)}" + (": the prompt plays like the strong vector, as far as the noise lets one "
                                          "tell. " if like_strong else ": the prompt changes play differently. "))
             + f"Pooled per trait × side: prompt vs vector r = {fmt(r_pv[0], 2)}, both vs vector {fmt(r_bv[0], 2)}, "
             f"both vs prompt {fmt(r_bp[0], 2)}. Mean |reward change| (SD): "
             + ", ".join(f"{a} {np.nanmean(np.abs(per[a])):.2f}" for a in ARMS) + ".")
    breaks = [f"{t} {'+' if s > 0 else '−'} ({res[(t, s, 2, 'vector')][0]:.0f}% vector vs "
              f"{res[(t, s, 2, 'prompt')][0]:.0f}% prompt)" for t, s in sides
              if res[(t, s, 2, "vector")][0] - res[(t, s, 2, "prompt")][0] > 5]
    f.append(f"<b>Cost:</b> {fb0:.1f}% of turns unparseable unsteered"
             + (f"; the vector breaks answers far more than the prompt for {'; '.join(breaks)}, and makes those "
                "answers many times longer (table e)." if breaks else "; no arm breaks answers."))
    ci = lambda a: float(np.mean([v for v in (al(t, s, 4, a) for t, s in sides) if math.isfinite(v)] or [float("nan")]))
    f.append(f"<b>Contagion</b> (co-players' readout the intended way, mean over traits and sides): vector "
             f"{ci('vector'):+.2f}, prompt {ci('prompt'):+.2f}, both {ci('both'):+.2f} SD.")
    if g_n:
        f.append(f"<b>Groups:</b> with everyone or the whole deceiving side steered, the prompt again moves the readout "
                 f"further than the vector at c = ±1 in {g_more} of {g_n} cases (table h).")
    take = [f"<b>Prompt, vector, both.</b> At matched dose the prompt moves the language more than the vector "
            f"({len(stronger)} of {len(sides)} trait × side cases, {len(p_beats)} of {len(games)} games) and both "
            f"together most ({len(both_top)} of {len(sides)})"
            + (f"; but the {wrong[0].split(' (')[0]} prompt fails where the vector works" if wrong else "")
            + (f"; suppressing {', '.join(b.split(' (')[0] for b in breaks[:2])} with the vector breaks answers"
               if breaks else "")
            + (f"; in play they agree about as well as the vector with itself at another strength (r = {fmt(gp[0], 2)} "
               f"vs {fmt(gv[0], 2)})" if same_as_noise else
               (f"; in play the prompt acts like the strong vector (c = ±2: r = {fmt(gp2[0], 2)}, about the "
                f"{fmt(gv[0], 2)} the vector reaches with itself), not the matched c = ±1 ({fmt(gp[0], 2)})" if like_strong
                else f"; in play the prompt differs from the vector (r = {fmt(gp[0], 2)} at c = ±1, {fmt(gp2[0], 2)} at "
                f"±2; vector vs itself {fmt(gv[0], 2)})"))
            + (f"; together they are more than the sum on the readout for {len(syn_r)} of {len(sides)} cases, less for "
               f"{len(sub_r)}" if syn_r or sub_r else "; together they simply add")
            + " (section 19)."]
    legend = ("<div class=legend><span><i style='background:var(--pos)'></i>vector</span><span><i style='background:"
              "var(--s3)'></i>prompt + vector (both)</span><span><i style='background:var(--neg);height:7px;width:7px'>"
              "</i>prompt (+ / − instruction, shown at c = ±1)</span><span>bars: 95% CI</span></div>")
    body = (legend + "<h3>a. Over the dose: readout of the trait</h3><div class=charts>" + "".join(charts["readout"])
            + "</div>" + g19 + "<details><summary>a. Over the dose: the steered agent's reward change (SD, all games)"
            "</summary><div class=charts>" + "".join(charts["reward"]) + "</div>" + g19 + "</details>" + "".join(tabs)
            + "".join(gtab)
            + "".join(ttab) + "".join(stab))
    j_f, j_html = prompt_outcomes(rows, base, scale)
    f += j_f
    body += j_html
    return take, section(19, "Vector, prompt or both? Side by side",
                         "Giving an agent a persona by the vector, by the instruction in its system prompt, or by both: "
                         "which moves its language more, which changes its play, which spreads to the others, which "
                         "costs answer quality? The same for everyone steered and for the whole deceiving side.",
                         "One steered agent per episode (conditions one, prompt_one, both_one). The prompt has no "
                         "strength, so it is compared with the vector and with both at c = ±1 (+ = the trait's "
                         "instruction / c &gt; 0, − = its opposite / c &lt; 0); the vector at c = ±2 for reference. "
                         "Each a mean over episodes with a 95% CI (hover): <b>b</b> the agent's readout of its trait "
                         "(SD vs unsteered agents of the same game); <b>c</b> its reward vs the same seat in the "
                         "unsteered episode with the same seed (SD units); <b>d</b> unparseable answers; <b>e</b> answer "
                         "length vs the game's unsteered answers; <b>f</b> the unsteered co-players' readout of the "
                         "trait; <b>g</b> per game; <b>h</b> everyone steered (all vs prompt_all) and the whole "
                         "deceiving side (evil_team vs evil_team_prompt); <b>i</b> synergy: both − (vector + prompt), all three "
                         "against the same unsteered reference (0 = they add); <b>j</b> outcomes: significant gains and "
                         "losses per arm, prompt vs vector head to head per game × trait × side, for one agent's own "
                         "result and for the group when everyone is steered. Charts a: all coefficients, all games pooled.",
                         "Compare the three arms within a row. Readout: + rows should be positive, − rows negative; a "
                         "prompt cell with the wrong sign = the instruction backfires. 'both ÷ (vector + prompt)' ≈ 1: "
                         "the two add; &gt; 1 reinforce; &lt; 1 overlap. Reward: all arms share the same unsteered "
                         "reference, so differences between arms are more reliable than each value's distance from 0. "
                         "Section 6 adds the direction (fingerprint), visibility and the vector strength the prompt "
                         "equals.", f, body)


MODES = {"text": "var(--s1)", "image": "var(--s2)"}  # observation mode -> chart colour
TRAIT_COLOURS = [f"var(--s{i})" for i in range(1, 9)]


def _line_fit(xs, ys):
    """(intercept, slope) of the least-squares line through the points, for drawing (None if it is undefined)."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0:
        return None
    b, a = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def _diff_test(a, b):
    """b - a for two independent estimates {"b", "se"}, with SE and two-sided p."""
    if not (a and b and math.isfinite(a.get("b", float("nan"))) and math.isfinite(b.get("b", float("nan")))):
        return {"b": float("nan"), "se": float("nan"), "p": float("nan"), "n": 0}
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2) if math.isfinite(a["se"]) and math.isfinite(b["se"]) else float("nan")
    d = b["b"] - a["b"]
    return {"b": d, "se": se, "n": a.get("n", 0) + b.get("n", 0),
            "p": math.erfc(abs(d / se) / math.sqrt(2)) if math.isfinite(se) and se > 0 else float("nan")}


NO_FIT = {"b": float("nan"), "se": float("nan"), "p": float("nan"), "n": 0}


def synergy_cells(rows, traits, base, scale):
    """Per (game cell, trait, side, kind): both − (vector + prompt) for one steered agent, the vector and both at
    c = ±1, all three against the same unsteered reference. kind 'readout': the trait's readout the intended way, in
    SD of the game's unsteered agents; 'readout_raw': the same in raw projection units (comparable across games,
    whose unsteered spread differs a hundredfold); 'reward': the paired reward change in SD units."""
    conds = ("one", "prompt_one", "both_one")
    pick = lambda r: r["steered"] and r["condition"] in conds and not r["random"]
    key = lambda r: (r["cell"], ARM_OF[r["condition"]], r["trait"])
    arm, rew = _by([r for r in rows if pick(r)], key), _by(paired(rows, base, pick), key)
    # the raw readout as a change against the game's unsteered agents: uncentred, the unsteered level would not cancel
    # in both − (vector + prompt) and would stand in for a synergy
    level = defaultdict(list)
    for r in rows:
        if r["condition"] == "none":
            for t, v in r.get("mon_raw", {}).items():
                level[(r["cell"], t)].append(v)
    level = {k: float(np.mean(v)) for k, v in level.items()}
    raw = lambda t: lambda r: (r["mon_raw"][t] - level[(r["cell"], t)]
                               if t in r.get("mon_raw", {}) and (r["cell"], t) in level else None)
    value = {"readout": lambda t: lambda r: r["mon"].get(t), "readout_raw": raw,
             "reward": lambda t: lambda r: r["diff"] / scale.get(r["cell"], 1.0)}
    out = {}
    for cell in sorted({k[0] for k in arm}):
        for t in traits:
            for s in (1, -1):
                for kind, val in value.items():
                    src = rew if kind == "reward" else arm
                    parts = [_arm_point(src.get((cell, a, t), []), val(t), s, None if a == "prompt" else 1.0)
                             for a in ("both", "vector", "prompt")]
                    if any(n < 2 for _, _, n in parts):
                        continue
                    al = 1 if kind == "reward" else s
                    d = al * (parts[0][0] - parts[1][0] - parts[2][0])
                    se = math.sqrt(sum(x[1] ** 2 for x in parts))
                    if math.isfinite(d) and math.isfinite(se) and se > 0:
                        out[(cell, t, s, kind)] = {"b": d, "se": se, "n": min(n for _, _, n in parts),
                                                   "clusters": sum(n for _, _, n in parts),
                                                   "p": math.erfc(abs(d / se) / math.sqrt(2))}
    return out


def random_effects(estimates):
    """DerSimonian–Laird pool of per-game estimates [{"b", "se"}]: (mean, SE, tau²). The SE includes how much the
    games differ (tau²), so a pooled verdict is not overconfident when the games disagree."""
    v = [(e["b"], e["se"]) for e in estimates if math.isfinite(e["b"]) and math.isfinite(e["se"]) and e["se"] > 0]
    if len(v) < 2:
        return (v[0][0], v[0][1], 0.0) if v else (float("nan"), float("nan"), float("nan"))
    y, w = np.array([b for b, _ in v]), np.array([1 / se ** 2 for _, se in v])
    mu = float((w * y).sum() / w.sum())
    q = float((w * (y - mu) ** 2).sum())
    tau2 = max(0.0, (q - (len(v) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (1 / w + tau2)
    return float((ws * y).sum() / ws.sum()), float(math.sqrt(1 / ws.sum())), tau2


def heterogeneity(syn, kind):
    """Per (observation mode, trait, side): do the games' synergies differ beyond their SEs? Cochran's Q over the games
    with an estimate. {(mode, t, s): {"Q", "games", "p", "pooled"}}."""
    by = defaultdict(list)
    for (cell, t, s, k), v in syn.items():
        if k == kind:
            by[(cell[1], t, s)].append(v)
    out = {}
    for key, vs in by.items():
        if len(vs) < 3:
            continue
        w = np.array([1 / v["se"] ** 2 for v in vs])
        y = np.array([v["b"] for v in vs])
        mu = float((w * y).sum() / w.sum())
        q = float((w * (y - mu) ** 2).sum())
        out[key] = {"Q": q, "games": len(vs), "pooled": mu, "p": chi2_sf(q, len(vs) - 1)}
    return out


def modality_channel_section(rows, traits, base, scale, steered):
    """25: observation mode x steering channel. Does seeing the game as a picture change the slopes (language and
    result) game by game; does it change what the vector, the prompt and both do; and do vector and prompt add up,
    in the language and in the result, in each mode?"""
    modes = defaultdict(set)
    for r in rows:
        modes[r["group"]].add(r["obs"])
    groups = sorted(g for g, m in modes.items() if {"text", "image"} <= m)
    title = "Image vs text × vector vs prompt: slopes and synergy"
    if not groups:
        return [], section(25, title, "", "", "", [], "")
    tcol = {t: TRAIT_COLOURS[i % len(TRAIT_COLOURS)] for i, t in enumerate(traits)}
    vec = lambda r: is_vector(r) and r["team_cond"] is None
    rd_of = _by([r for r in rows if r["group"] in groups and (vec(r) or r["condition"] == "none")], lambda r: r["cell"])
    rw_of = _by([r for r in steered if r["group"] in groups and math.isfinite(r["diff"])],
                lambda r: (r["cell"], r["trait"]))
    sd = lambda r: r["diff"] / scale.get(r["cell"], 1.0)
    kinds = ("readout", "reward")

    # a/b. per game x trait: the readout and reward slope in each mode, and the curves they summarise
    slopes, curves = {}, {}
    for g in groups:
        charts = []
        for t in traits:
            for kind in kinds:
                series, fits, notes = {}, {}, {}
                for m in MODES:
                    cell = (g, m)
                    if kind == "readout":
                        pts = [r for r in rd_of.get(cell, []) if t in r["mon"]
                               and (r["condition"] == "none" or r["trait"] == t)]
                        x = lambda r: r["coef"] if r["condition"] != "none" else 0.0
                        y = lambda r: r["mon"][t]
                        fit = readout_slope(pts, t, vec, by_cell=False) \
                            if any(r["condition"] != "none" for r in pts) else NO_FIT
                    else:
                        pts = rw_of.get((cell, t), [])
                        x, y = (lambda r: r["coef"]), sd
                        fit = _diff_slope(pts, scale) if pts else NO_FIT
                    slopes[(g, t, kind, m)] = fit
                    if not any(r["condition"] != "none" for r in pts):
                        continue
                    series[m] = [(c, *mean_ci([y(r) for r in rs], [r["cluster"] for r in rs])[:2])
                                 for c, rs in sorted(_by(pts, x).items())] \
                        + ([] if kind == "readout" else [(0.0, 0.0, float("nan"))])
                    fits[m] = _line_fit([x(r) for r in pts], [y(r) for r in pts])
                    notes[m] = (f"{m}: slope {fit['b']:+.2f} ± {1.96 * fit['se']:.2f} per unit c"
                                if math.isfinite(fit.get("b", float("nan"))) else m)
                if series:
                    ylab = f"{t} readout (SD)" if kind == "readout" else "reward change (SD)"
                    charts.append(f"<div class=chart><h3>{html.escape(t)}: "
                                  f"{'language' if kind == 'readout' else 'result'}</h3>"
                                  f"{line_chart(series, f'{g} {t} {kind}', ylab, colours=MODES, notes=notes, fits=fits)}"
                                  "</div>")
        curves[g] = "".join(charts)
    diffs = {(g, t, k): _diff_test(slopes[(g, t, k, "text")], slopes[(g, t, k, "image")])
             for g in groups for t in traits for k in kinds}
    dq = dict(zip(diffs, bh([v["p"] for v in diffs.values()])))
    scat, agree = [], {}
    for k in kinds:
        pts = [(slopes[(g, t, k, "text")]["b"], slopes[(g, t, k, "image")]["b"], f"{g} × {t}")
               for g in groups for t in traits]
        pts = [p for p in pts if math.isfinite(p[0]) and math.isfinite(p[1])]
        r_ = pearson([p[0] for p in pts], [p[1] for p in pts])
        same = sum(1 for p in pts if p[0] * p[1] > 0)
        moved = [(g, t) for g in groups for t in traits if sig(diffs[(g, t, k)], dq[(g, t, k)])]
        agree[k] = (r_, same, len(pts), moved)
        lab = "readout slope (SD per unit c)" if k == "readout" else "reward slope (SD per unit c)"
        scat.append(f"<div class=chart><h3>{'Language' if k == 'readout' else 'Result'}: slope with images vs with "
                    f"text</h3>{scatter_chart(pts, f'text: {lab}', f'image: {lab}', colour=lambda l: tcol[l.split(' × ')[1]])}"
                    f"<p class=muted>r = {fmt(r_[0], 2)} [{fmt(r_[1], 2)}, {fmt(r_[2], 2)}], same sign in {same} of "
                    f"{len(pts)}; image ≠ text (BH) in {len(moved)}</p></div>")
    trait_legend = ("<div class=legend>" + "".join(f"<span><i style='background:{c};height:7px;width:7px'></i>"
                                                    f"{html.escape(t)}</span>" for t, c in tcol.items())
                    + "<span>one dot = one game × trait; dashed: image = text</span></div>")

    # c/d. channel x mode: vector (c = ±1), prompt, both, per mode, readout and reward; and their synergy
    at = lambda a: None if a == "prompt" else 1.0
    conds = ("one", "prompt_one", "both_one")
    pick = lambda r: r["steered"] and r["condition"] in conds and not r["random"] and r["group"] in groups
    arm = _by([r for r in rows if pick(r)], lambda r: (r["obs"], ARM_OF[r["condition"]], r["trait"]))
    rew = _by(paired(rows, base, pick), lambda r: (r["obs"], ARM_OF[r["condition"]], r["trait"]))
    res = {}
    for m in MODES:
        for t in traits:
            for s in (1, -1):
                for a in ARMS:
                    res[(m, t, s, "readout", a)] = _arm_point(arm.get((m, a, t), []), lambda r, t=t: r["mon"].get(t),
                                                              s, at(a))
                    res[(m, t, s, "reward", a)] = _arm_point(rew.get((m, a, t), []), sd, s, at(a))
    sides = [(t, s) for t in traits for s in (1, -1)]
    label = lambda t, s: f"{t} {'+' if s > 0 else '−'}"
    align = lambda k, s: s if k == "readout" else 1  # readout: the intended way; reward: as it is
    arm_charts = {}
    for k in kinds:
        items = {m: [(label(t, s), {a: (align(k, s) * res[(m, t, s, k, a)][0], 1.96 * res[(m, t, s, k, a)][1])
                                    for a in ARMS}) for t, s in sides] for m in MODES}
        ends = [v + e * (h if math.isfinite(h) else 0) for its in items.values() for _, ser in its
                for v, h in ser.values() if math.isfinite(v) for e in (-1, 1)]
        span = (min(ends + [0]), max(ends + [0])) if ends else None
        xl = "readout of the trait, intended way (SD)" if k == "readout" else "reward change (SD)"
        arm_charts[k] = "".join(f"<div class=chart><h3>{m}</h3>{forest_chart(items[m], ARMS, xl, span=span)}</div>"
                                for m in MODES)
    syn = {}
    for m in MODES:
        for t, s in sides:
            for k in kinds:
                (mb, sb, nb), (mv, sv, nv), (mp, sp, np_) = (res[(m, t, s, k, a)] for a in ("both", "vector", "prompt"))
                d, se = align(k, s) * (mb - mv - mp), math.sqrt(sb ** 2 + sv ** 2 + sp ** 2)
                syn[(m, t, s, k)] = {"b": d, "se": se, "n": nb + nv + np_,
                                     "p": math.erfc(abs(d / se) / math.sqrt(2)) if math.isfinite(se) and se > 0
                                     else float("nan")}
    sq = dict(zip(syn, bh([v["p"] for v in syn.values()])))
    syn_charts = {k: forest_chart([(label(t, s), {m: (syn[(m, t, s, k)]["b"], 1.96 * syn[(m, t, s, k)]["se"])
                                                  for m in MODES}) for t, s in sides], MODES,
                                  "both − (vector + prompt)" + (", intended way (SD)" if k == "readout" else " (SD)"))
                  for k in kinds}

    # e. per game: pooling averages over games, so synergy in one and interference in another can cancel out
    per = synergy_cells(rows, traits, base, scale)
    pcells = sorted({k[0] for k in per})
    cols_ = [label(t, s) for t, s in sides]
    unlabel = {label(t, s): (t, s) for t, s in sides}
    game_tabs, game_found = [], {}
    for k, title_ in (("readout", "Language: both − (vector + prompt), intended way (SD of the game's unsteered agents)"),
                      ("reward", "Result: both − (vector + prompt) (SD)")):
        tab, found, q = matrix(title_, "game", pcells, cols_, lambda c, col, k=k: per.get((c, *unlabel[col], k)))
        game_tabs.append(tab)
        game_found[k] = [(key, v) for key, v in found.items() if sig(v, q[key])]
    het = {k: heterogeneity(per, "readout_raw" if k == "readout" else k) for k in kinds}
    het_q = {k: dict(zip(het[k], bh([v["p"] for v in het[k].values()]))) for k in kinds}
    n_cells = {k: sum(1 for key in per if key[3] == k) for k in kinds}

    def pooled(m, a):
        """Mean over traits x sides of how far arm a moves the trait the intended way in mode m, with its SE."""
        v = [(s * res[(m, t, s, "readout", a)][0], res[(m, t, s, "readout", a)][1]) for t, s in sides]
        v = [(x, e) for x, e in v if math.isfinite(x) and math.isfinite(e)]
        if not v:
            return float("nan"), float("nan")
        return float(np.mean([x for x, _ in v])), math.sqrt(sum(e ** 2 for _, e in v)) / len(v)
    ptab = ["<h3>Pooled: how far each channel moves the trait the intended way (SD, mean over traits × sides)</h3>"
            "<div class=wrap><table><tr><th class=l>channel</th><th>text</th><th>image</th><th>image − text</th></tr>"]
    pool = {}
    for a in ARMS:
        (mt, et), (mi, ei) = pooled("text", a), pooled("image", a)
        pool[a] = {"text": mt, "image": mi, "diff": _diff_test({"b": mt, "se": et}, {"b": mi, "se": ei})}
        ptab.append(f"<tr><td class=l>{a}{'' if a == 'prompt' else ' (c = ±1)'}</td>"
                    + _value_td(mt, 1.96 * et, scale=10) + _value_td(mi, 1.96 * ei, scale=10)
                    + slope_cell(pool[a]["diff"]) + "</tr>")
    ptab.append("</table></div>" + games_note({r["cell"] for r in rows if pick(r)}))
    inter = _diff_test(pool["vector"]["diff"], pool["prompt"]["diff"])  # does the picture change the prompt more?

    # findings
    f = []
    for k, what in (("readout", "language (readout)"), ("reward", "result (reward)")):
        r_, same, n, moved = agree[k]
        f.append(f"<b>Slopes, image vs text, {what}:</b> per game × trait the slope with images correlates "
                 f"r = {fmt(r_[0], 2)} with the slope with text (n = {n}); same sign in {same} of {n}; significantly "
                 f"different (BH) in {len(moved)}"
                 + (": " + ", ".join(f"{g} × {t} ({slopes[(g, t, k, 'text')]['b']:+.2f} with text, "
                                     f"{slopes[(g, t, k, 'image')]['b']:+.2f} with images)" for g, t in moved[:5])
                    + (f"; {len(moved) - 5} more" if len(moved) > 5 else "") if moved else "") + ".")
    f.append("<b>Channels by mode</b> (pooled, intended way, SD): "
             + "; ".join(f"{a} {fmt(pool[a]['text'], 2)} with text, {fmt(pool[a]['image'], 2)} with images" for a in ARMS)
             + (f". The picture changes the prompt's effect by {inter['b']:+.2f} SD more than the vector's "
                f"(± {fmt(1.96 * inter['se'], 2)}, p = {fmt(inter['p'], 3)})" if math.isfinite(inter["b"]) else "") + ".")
    n_syn = {}
    for k, what in (("readout", "language"), ("reward", "result")):
        hit = lambda m, sign: [(t, s) for t, s in sides if sig(syn[(m, t, s, k)], sq[(m, t, s, k)])
                               and sign * syn[(m, t, s, k)]["b"] > 0]
        names = {sign: [f"{label(t, s)} with {m} ({syn[(m, t, s, k)]['b']:+.2f})" for m in MODES
                        for t, s in hit(m, sign)] for sign in (1, -1)}
        n_syn[k] = len(names[1]) + len(names[-1])
        listed = lambda v: ", ".join(v[:5]) + (f" and {len(v) - 5} more" if len(v) > 5 else "")
        f.append(f"<b>Synergy, {what}:</b> prompt + vector is more than the sum of the two in "
                 f"{len(hit('text', 1))} (text) / {len(hit('image', 1))} (images) of {len(sides)} trait × side cases, "
                 f"less in {len(hit('text', -1))} / {len(hit('image', -1))}"
                 + ((f". More: {listed(names[1])}." if names[1] else "")
                    + (f" Less: {listed(names[-1])}." if names[-1] else "") + " Everywhere else they simply add."
                    if n_syn[k] else ": in both modes they simply add."))
    differ = {k: [key for key in het[k] if het_q[k][key] < 0.05] for k in kinds}
    for k, what in (("readout", "language"), ("reward", "result")):
        more = [key for key, v in game_found[k] if v["b"] > 0]
        less = [key for key, v in game_found[k] if v["b"] < 0]
        top = sorted(game_found[k], key=lambda kv: -abs(kv[1]["b"]))[:4]
        examples = ", ".join(f"{cell_name(key[0])} × {key[1]} ({v['b']:+.1f})" for key, v in top)
        f.append(f"<b>Per game, {what} (table e):</b> of {n_cells[k]} game × trait × side cells, prompt + vector is "
                 f"significantly more than the sum in {len(more)} and less in {len(less)} (BH over the table)"
                 + (f", e.g. {examples}" if top else "")
                 + (". Too few games per mode to test whether they differ." if not het[k] else
                    f". The games differ beyond their noise (Cochran's Q, BH) in {len(differ[k])} of {len(het[k])} "
                    "trait × side × mode cells"
                    + (" (language measured in raw projection units here, since the games' unsteered spread differs "
                       "widely)" if k == "readout" else "")
                    + (": the pooled verdict above is an average over games that disagree, not a property of the "
                       "persona." if len(differ[k]) > len(het[k]) / 4 else
                       ": the pooled verdict holds game by game, as far as the episodes per game allow.")))
    rr, rw_ = agree["readout"][0][0], agree["reward"][0][0]
    take = [f"<b>Image vs text, vector vs prompt.</b> Game by game, how strongly steering moves the language carries "
            f"over from text to images (slopes r = {fmt(rr, 2)}), how it moves the result "
            + ("much less" if math.isfinite(rr) and math.isfinite(rw_) and rw_ < rr - 0.2 else "about as well")
            + f" (r = {fmt(rw_, 2)}). The prompt moves the language {fmt(pool['prompt']['text'], 1)} SD with text and "
            f"{fmt(pool['prompt']['image'], 1)} with images, the vector (c = ±1) {fmt(pool['vector']['text'], 1)} and "
            f"{fmt(pool['vector']['image'], 1)}. On the result, prompt and vector "
            + (f"add up to more or less than their sum in {n_syn['reward']} of {2 * len(sides)} trait × side × mode cases"
               if n_syn["reward"] else "simply add in both modes")
            + (f" and in {n_cells['reward'] - len(game_found['reward'])} of {n_cells['reward']} game × trait × side "
               "cells" if n_cells["reward"] else "")
            + (f"; on the language, whether they reinforce or overlap depends on the game ({len(differ['readout'])} of "
               f"{len(het['readout'])} trait × side × mode cells differ between games)" if differ["readout"] else "")
            + " (section 25)."]
    g25 = games_note([(g, m) for g in groups for m in MODES], "Games in these charts")
    g25c = games_note({r["cell"] for r in rows if pick(r)}, "Games in these charts")
    body = (f"<h3>a. Slopes per game × trait: images vs text</h3>{trait_legend}<div class=charts>{''.join(scat)}</div>"
            + g25 +
            "<h3>b. The curves behind each slope, per game</h3><div class=legend><span><i style='background:var(--s1)'>"
            "</i>text</span><span><i style='background:var(--s2)'></i>image</span><span>dots: mean per c, bars: 95% CI; "
            "dashed: the fitted straight line (hover it for the slope)</span></div>"
            + "".join(f"<details><summary>{html.escape(g)}</summary><div class=charts>{curves[g]}</div></details>"
                      for g in groups)
            + "<h3>c. Vector, prompt, both: text and image side by side</h3><div class=legend>"
            + "".join(f"<span><i style='background:{c};height:7px;width:7px'></i>{a}</span>" for a, c in ARMS.items())
            + "<span>vector and both at c = ±1; bars: 95% CI</span></div>"
            f"<h3>Language: readout of the trait</h3><div class=charts>{arm_charts['readout']}</div>"
            f"<h3>Result: reward change</h3><div class=charts>{arm_charts['reward']}</div>{g25c}" + "".join(ptab)
            + "<h3>d. Synergy: both − (vector + prompt), text vs image</h3><div class=legend>"
            "<span><i style='background:var(--s1);height:7px;width:7px'></i>text</span><span><i style='background:"
            "var(--s2);height:7px;width:7px'></i>image</span><span>0 = they add; right of 0 = more than the sum (on the "
            "readout: further the intended way)</span></div>"
            f"<div class=charts><div class=chart><h3>Language</h3>{syn_charts['readout']}</div>"
            f"<div class=chart><h3>Result</h3>{syn_charts['reward']}</div></div>{g25c}"
            "<h3>e. Synergy per game</h3><p class=muted>Every game with all three arms, text and image separately. "
            "Cell: both − (vector + prompt); colour by how certain (red = more than the sum, blue = less), stars: "
            "Benjamini–Hochberg over the table; hover for n and SE. Language values are in each game's own SD units, "
            "which differ a hundredfold between games: compare colours and stars across games, not the numbers.</p>"
            + "".join(game_tabs))
    return take, section(25, title,
                         "Does seeing the game as a picture change what steering does, game by game? Does it change "
                         "the vector, the prompt and both alike? And do vector and prompt add up, in the language and in "
                         "the result, with text and with images?",
                         "Games played both ways only (" + ", ".join(groups) + "). <b>a</b>: per game × trait the slope "
                         "per unit c of the steered agent's readout of its trait (SD vs the game's unsteered agents) and "
                         "of its reward change (same seat, same seed, unsteered; SD units), with text (x) and with images "
                         "(y); persona vector, settings without a side. <b>b</b>: the dose curves those slopes "
                         "summarise, with the fitted line drawn. <b>c</b>: one steered agent, vector (c = ±1), prompt and "
                         "both (c = ±1) per mode, as in section 19. <b>d</b>: both − (vector + prompt) per mode, all three "
                         "against the same unsteered reference. <b>e</b>: the same per game (every game with all three "
                         "arms), and whether the games differ beyond their noise (Cochran's Q per trait × side × mode; "
                         "for the language on the raw projection, which is comparable across games). Tests: image − "
                         "text with independent SEs; Benjamini–Hochberg over each chart's or table's cells.",
                         "In a, dots on the dashed diagonal = the picture changes nothing; a tight cloud along it = "
                         "steering transfers across modes game by game. In b, the dashed line is the slope: a curve far "
                         "from its line (bends, plateaus) is summarised badly by one number (section 16). In d, 0 = the "
                         "two channels add; on the readout, values below 0 can be a ceiling of the projection rather "
                         "than competition.", f, body)


def length_section(rows, traits):
    """20: how much of the readout shift is the answer getting shorter or longer."""
    tab = ["<div class=wrap><table><tr><th class=l>trait</th><th>readout per unit c (SD)</th><th>answer length "
           "change per unit c</th><th>readout per unit c with answer length held fixed (SD)</th><th>share of the "
           "readout shift that is not answer length</th></tr>"]
    kept, out = {}, {}
    base_by = _by([r for r in rows if r["condition"] == "none" and r["chars"]], lambda r: r["cell"])
    for t in traits:
        pts = [r for r in rows if is_vector(r, t) and r["team_cond"] is None and r["chars"] and t in r["mon"]]
        pts += [r for c in {r["cell"] for r in pts} for r in base_by.get(c, []) if t in r["mon"]]
        x = [r["coef"] if r["condition"] != "none" else 0.0 for r in pts]
        y = [r["mon"][t] for r in pts]
        ll = [math.log(max(r["chars"], 1.0)) for r in pts]
        cl, fe = [r["cluster"] for r in pts], [r["cell"] for r in pts]
        plain = slope(x, y, cl, fe)
        held = ols(y, list(zip(x, ll)), cl, fe)[0]
        ln = slope(x, ll, cl, fe)
        share = held["b"] / plain["b"] if plain["b"] else float("nan")
        kept[t], out[t] = share, ln["b"]
        pct = 100 * (math.exp(ln["b"]) - 1) if math.isfinite(ln["b"]) else float("nan")
        tab.append(f"<tr><td class=l>{t}</td>{slope_cell(plain)}<td>{pct:+.0f}%</td>{slope_cell(held)}"
                   f"<td>{100 * share:.0f}%</td></tr>")
    tab.append("</table></div>" + games_note({r["cell"] for r in rows if is_vector(r) and r["team_cond"] is None
                                              and r["chars"]}))
    low = {t: v for t, v in kept.items() if math.isfinite(v) and v < 0.8}
    f = [f"Holding answer length fixed keeps ≥ 80% of the readout shift for {len(traits) - len(low)} of {len(traits)} traits."]
    if low:
        f.append("Partly length: " + "; ".join(f"{t} ({100 * v:.0f}% left; answers {100 * (math.exp(out[t]) - 1):+.0f}% "
                                                f"per unit c)" for t, v in low.items())
                 + " — for these, part of what reads as the trait is answers getting shorter or longer.")
    take = [f"<b>The readout is not just answer length.</b> Controlling for length keeps most of the shift for "
            f"{len(traits) - len(low)} of {len(traits)} traits"
            + (f"; for {', '.join(low)} about {100 * (1 - min(low.values())):.0f}% at most comes with the length change" if low else "")
            + " (section 20)."]
    return take, section(20, "Is the readout just answer length?",
                         "Steering changes how long answers are (section 13). Is the trait readout really measuring "
                         "the trait, or mostly that answers got shorter or longer?",
                         "Steered agents (persona vector) and unsteered agents at c = 0, as in section 2. The readout "
                         "slope over the coefficient, then the same slope in a regression that also contains the log of "
                         "the agent's mean answer length (game fixed effects, SEs clustered by episode). Length per "
                         "unit c: the change in answer length per unit coefficient.",
                         "If 100% of the shift remains with length held fixed, length explains none of it. A low share "
                         "means steering and the readout both move with length: the readout partly measures verbosity. "
                         "This is a statistical control, not proof — length can itself be part of a persona (an "
                         "assertive agent is brief).", f, "".join(tab))


def group_section(rows, traits, steered, scale):
    """21: does a group dilute one steered agent; is steering everyone the sum of steering one?"""
    by_cell = _by(rows, lambda r: r["cell"])
    own = _by([r for r in steered if r["condition"] == "one" and _single_vector(r)], lambda r: (r["cell"], r["trait"]))
    size = {}
    for cell, cr in by_cell.items():
        if cr and cr[0].get("players"):
            size[cell] = int(cr[0]["players"])
    per = []
    for (cell, t), rs in own.items():
        if len(rs) < 50 or cell not in size:
            continue
        s = _diff_slope(rs, scale)
        co = [r for r in by_cell[cell] if r["condition"] == "one" and not r["steered"] and r["trait"] == t and t in r["mon"]]
        b0 = [r for r in by_cell[cell] if r["condition"] == "none" and t in r["mon"]]
        k = slope([r["coef"] for r in co] + [0.0] * len(b0), [r["mon"][t] for r in co + b0], [r["cluster"] for r in co + b0])
        per.append((size[cell], abs(s["b"]), k["b"]))
    by_n = _by(per, lambda p: p[0])
    tab = ["<h3>One steered agent, by group size</h3><div class=wrap><table><tr><th class=l>players per game</th>"
           "<th>game × trait cells</th>"
           "<th>median |own reward slope| (SD per unit c)</th><th>median contagion (co-players' readout, SD per unit c)</th></tr>"]
    for n, ps in sorted(by_n.items()):
        tab.append(f"<tr><td class=l>{n}</td><td>{len(ps)}</td><td>{statistics.median(p[1] for p in ps):.3f}</td>"
                   f"<td>{statistics.median(p[2] for p in ps):+.3f}</td></tr>")
    tab.append("</table></div>" + games_note({cell for (cell, t), rs in own.items() if len(rs) >= 50 and cell in size}))
    r_own = _rank_corr([p[0] for p in per], [p[1] for p in per])
    r_con = _rank_corr([p[0] for p in per], [p[2] for p in per])
    roles = role_cells(rows)  # their total reward only says which side won (section 22)
    wel = welfare([r for r in rows if r["condition"] in ("one", "all") and r["via"] == "vector" and not r["random"]
                   and r["team_cond"] is None and r["cell"] not in roles],
                  [r for r in rows if r["condition"] == "none" and r["cell"] not in roles])
    wb = _by(wel, lambda r: (r["cell"], r["trait"], r["condition"]))
    pairs = []
    for cell in sorted(size):
        for t in traits:
            a, o = wb.get((cell, t, "all"), []), wb.get((cell, t, "one"), [])
            if len(a) >= 40 and len(o) >= 40:
                pairs.append((cell, t, _diff_slope(o, None), _diff_slope(a, None)))
    q = bh([p[3]["p"] for p in pairs] + [p[2]["p"] for p in pairs])
    qa, qo = q[:len(pairs)], q[len(pairs):]
    x = np.array([p[2]["b"] for p in pairs]); y = np.array([p[3]["b"] for p in pairs])
    ok = np.isfinite(x) & np.isfinite(y)
    k_one = float((x[ok] @ y[ok]) / (x[ok] @ x[ok])) if ok.any() and (x[ok] @ x[ok]) > 0 else float("nan")
    nx = np.array([size[p[0]] * p[2]["b"] for p in pairs])
    k_n = float((nx[ok] @ y[ok]) / (nx[ok] @ nx[ok])) if ok.any() and (nx[ok] @ nx[ok]) > 0 else float("nan")
    flips = [(p, qa[i]) for i, p in enumerate(pairs) if sig(p[3], qa[i]) and (not sig(p[2], qo[i]) or
             np.sign(p[2]["b"]) != np.sign(p[3]["b"])) and abs(p[3]["b"]) > 3 * abs(p[2]["b"])]
    tab.append("<h3>Group welfare: everyone steered vs one steered (SD per unit c)</h3><div class=wrap><table><tr>"
               "<th class=l>game</th><th>trait</th><th>players</th><th>one agent steered: welfare per unit c</th>"
               "<th>players × that (if each steered agent added its own effect)</th><th>everyone steered: welfare per "
               "unit c</th></tr>")
    for i, (cell, t, o, a) in enumerate(pairs):
        if sig(a, qa[i]):
            tab.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td><td>{t}</td><td>{size[cell]}</td>"
                       f"{slope_cell(o, qo[i])}{_value_td(size[cell] * o['b'], None, scale=1, d=2)}{slope_cell(a, qa[i])}</tr>")
    tab.append("</table></div>")
    p_rank = lambda r: math.erfc(abs(r) * math.sqrt(max(len(per) - 2, 1) / max(1 - r * r, 1e-9)) / math.sqrt(2))
    trend = lambda r, what, down: (f"{down if r < 0 else 'grows'} with group size (rank correlation {r:+.2f}, "
                                   f"p = {p_rank(r):.3f})" if p_rank(r) < 0.05 else
                                   f"does not change measurably with group size (rank correlation {r:+.2f}, "
                                   f"p = {p_rank(r):.2f})")
    f = [f"One steered agent's own reward effect {trend(r_own, 'own', 'shrinks')}; its contagion "
         f"{trend(r_con, 'contagion', 'weakens')} ({len(per)} game × trait cells).",
         f"Steering everyone changes group welfare by about {k_one:.2f}× the effect of steering one agent. If each "
         f"steered agent added its own effect, it would be players × as much; it is {k_n:.2f} of that: effects do not "
         f"add up across agents ({len(pairs)} game × trait cells)."]
    if flips:
        f.append("Where everyone steered does something one steered agent does not (all-steered effect > 3× the one-agent "
                 "effect, and that one is not significant or has the other sign): "
                 + "; ".join(f"{cell_name(p[0])} × {p[1]} (one {p[2]['b']:+.2f}, all {p[3]['b']:+.2f})" for p, _ in flips[:8])
                 + ". A persona everyone shares can change the group's result where one agent with it does not.")
    take = [f"<b>Steering everyone is not the sum of steering one.</b> The group's welfare moves about {k_one:.1f}× as much "
            f"as with one steered agent, far from additive"
            + (f"; in {len(flips)} game × trait cells the all-steered group changes where one steered agent does not (e.g. "
               f"{cell_name(flips[0][0][0])} × {flips[0][0][1]})" if flips else "")
            + f". One agent's own effect {trend(r_own, 'own', 'shrinks').split(' (')[0]} (section 21)."]
    return take, section(21, "Group size and collective steering",
                         "Does one steered agent matter less in a bigger group? And if every agent is steered, does the "
                         "group change n times as much as with one?",
                         "Top: per game × trait of the one-agent condition, the steered agent's reward slope (as in "
                         "section 9, SD units) and the co-players' readout slope (contagion, as in section 8), by the "
                         "game's number of players. Bottom (games without hidden roles, where the total only says "
                         "which side won): group welfare (everyone's total reward vs the unsteered "
                         "episode with the same seed, as in section 7) per unit c, with one agent steered and with all "
                         "steered; 'players × one' is what all-steered would give if each agent's effect simply added. "
                         "The factors are least-squares ratios over all game × trait cells.",
                         "Factor ≈ players: effects add. ≈ 1: steering everyone does about what steering one does. "
                         "Rows of the bottom table: cells where steering everyone significantly changes welfare.",
                         f, "".join(tab))


ROLE_SETTINGS = ("evil_team", "evil_one", "good_team", "good_one", "all", "one")  # vector settings compared in hidden-role games


def deceiver_wins(rows):
    """One row per episode of a hidden-role game: did the deceiving side win (any of its agents rewarded)?"""
    eps = defaultdict(list)
    for r in rows:
        if r["team"] == "evil" and r["reward"] is not None:
            eps[r["cluster"]].append(r)
    return [{**rs[0], "won": 100.0 * any(r["reward"] > 0 for r in rs)} for rs in eps.values()]


def _two_props(a, b):
    """Difference of two win rates (in %) with a normal-approximation p: a, b = (mean %, SE, n)."""
    d, se = a[0] - b[0], math.hypot(a[1], b[1])
    return {"b": d, "se": se, "n": a[2] + b[2], "clusters": a[2] + b[2],
            "p": math.erfc(abs(d / se) / math.sqrt(2)) if math.isfinite(se) and se > 0 else float("nan")}


def role_section(rows, traits, base, scale):
    """22: hidden-role games: who carries the persona (the whole deceiving side, one deceiver, one honest player,
    everyone) and what it does to the deceivers' win rate, across games; then the same persona on opposite roles."""
    wins = deceiver_wins(rows)
    cells_ = sorted({w["cell"] for w in wins if w["condition"] in ROLE_SETTINGS})
    if not cells_:
        return [], section(22, "Hidden-role games: who carries the persona", "", "", "", [], "")
    colour = {c: f"var(--s{SIDE_SETTINGS.index(c) + 1})" for c in ROLE_SETTINGS}
    by = _by(wins, lambda w: (w["cell"], w["condition"]))
    rate = lambda ws, keep=lambda w: True: _episode_mean([w["won"] for w in ws if keep(w)],
                                                         [w["cluster"] for w in ws if keep(w)])
    side = lambda s, strong=None: lambda w: (w["coef"] or 0) * s > 0 and (
        strong is None or (abs(w["coef"]) > 1) == strong)

    # a. win rate over c per game, pooled over traits, one line per setting
    charts, base_txt = [], []
    for cell in cells_:
        b0 = rate(by.get((cell, "none"), []))
        base_txt.append(f"{cell_name(cell)} {b0[0]:.0f}% (n = {b0[2]})" if b0[2] else f"{cell_name(cell)}: no unsteered episodes")
        series = {}
        for c in ROLE_SETTINGS:
            ws = by.get((cell, c), [])
            cs = sorted({w["coef"] for w in ws if w["coef"]})
            if cs:
                series[c] = [(x, m, 1.96 * se) for x in cs for m, se, _ in [rate(ws, lambda w, x=x: w["coef"] == x)]]
        pr = by.get((cell, "evil_team_prompt"), [])
        if pr:
            series["prompt"] = [(s, m, 1.96 * se) for s in (-1.0, 1.0) for m, se, _ in [rate(pr, side(s))]]
        if b0[2]:
            series["unsteered"] = [(0.0, b0[0], 1.96 * b0[1])]
        charts.append(f"<div class=chart><h3>{html.escape(cell_name(cell))}</h3>"
                      f"{line_chart(series, cell_name(cell), 'deceivers win (%)', colours={**colour, 'unsteered': 'var(--rnd)'})}</div>")
    legend = ("<div class=legend>" + "".join(f'<span><i style="background:{colour[c]}"></i>{c}</span>' for c in ROLE_SETTINGS)
              + "<span><i style='background:var(--neg);height:7px;width:7px'></i>evil_team_prompt (+ / − instruction)</span>"
              "<span><i style='background:var(--rnd)'></i>unsteered</span><span>bars: 95% CI; pooled over the 8 traits</span></div>")

    # b. up vs down: does pushing the trait up rather than down help the deceivers? per game x setting x trait
    ud, keys = {}, []
    for cell in cells_:
        for c in ROLE_SETTINGS + ("evil_team_prompt",):
            for t in traits:
                ws = [w for w in by.get((cell, c), []) if w["trait"] == t]
                if ws:
                    ud[(cell, c, t)] = _two_props(rate(ws, side(1)), rate(ws, side(-1)))
    # c. strong vs mild per side, pooled over traits: does steering the deceivers hard break them either way?
    sm = {}
    for cell in cells_:
        for c in ROLE_SETTINGS:
            ws = by.get((cell, c), [])
            for s in (-1, 1):
                if ws:
                    sm[(cell, c, s)] = _two_props(rate(ws, side(s, True)), rate(ws, side(s, False)))
    # d. whole side vs one member / deceiver vs honest player, same trait and side
    cmp_ = {}
    for cell in cells_:
        for t in traits:
            for s in (-1, 1):
                r_ = {c: rate([w for w in by.get((cell, c), []) if w["trait"] == t], side(s)) for c in ROLE_SETTINGS}
                cmp_[(cell, t, s, "team − one deceiver")] = _two_props(r_["evil_team"], r_["evil_one"])
                cmp_[(cell, t, s, "team − one honest")] = _two_props(r_["good_team"], r_["good_one"])
                cmp_[(cell, t, s, "one deceiver − one honest")] = _two_props(r_["evil_one"], r_["good_one"])
    allk = [("ud", k) for k in ud] + [("sm", k) for k in sm] + [("cmp", k) for k in cmp_]
    src = {"ud": ud, "sm": sm, "cmp": cmp_}
    qs = dict(zip(allk, bh([src[a][k]["p"] for a, k in allk])))
    q_of = lambda a, k: qs[(a, k)]

    out = [legend, "<h3>a. Deceivers' win rate over c, per game (all traits pooled)</h3><div class=charts>",
           "".join(charts), "</div>"]
    cols = [(cell, c) for c in ("evil_team", "evil_one", "good_team", "good_one", "all", "evil_team_prompt") for cell in cells_
            if any((cell, c, t) in ud for t in traits)]
    tab = ["<h3>b. Pushed up minus pushed down: change in the deceivers' win rate (percentage points)</h3>"
           "<div class=wrap><table><tr><th class=l>trait</th>"
           + "".join(f"<th>{html.escape(c)}<br><span class=muted>{html.escape(cell_name(cell))}</span></th>"
                     for cell, c in cols) + "</tr>"]
    for t in traits:
        tds = []
        for cell, c in cols:
            d = ud.get((cell, c, t))
            if not d or not math.isfinite(d["b"]):
                tds.append("<td class=muted>–</td>")
                continue
            q = q_of("ud", (cell, c, t))
            tds.append(f"<td{heat(d['b'], d['se'])} title='up − down {d['b']:+.0f} ± {1.96 * d['se']:.0f} points, "
                       f"q={fmt(q, 4)}'>{d['b']:+.0f}<span class=sig>{stars(q)}</span></td>")
        tab.append(f"<tr><td class=l>{t}</td>{''.join(tds)}</tr>")
    tab.append("</table></div><p class=muted>Positive = the deceivers win more when the trait is pushed up (c &gt; 0) "
               "than when it is pushed down (c &lt; 0); for good_one and good_team the steered agents are honest players, so "
               "positive means the honest side loses more when it carries the trait. No unsteered reference needed: both halves "
               "come from the same sweep.</p>")
    out += tab
    smt = ["<h3>c. Strong (|c| &gt; 1) minus mild (|c| ≤ 1): change in the deceivers' win rate (points), all traits pooled</h3>"
           "<div class=wrap><table><tr><th class=l>game</th>"
           + "".join(f"<th>{c}: down</th><th>{c}: up</th>" for c in ROLE_SETTINGS) + "</tr>"]
    for cell in cells_:
        tds = ""
        for c in ROLE_SETTINGS:
            for s in (-1, 1):
                d = sm.get((cell, c, s))
                tds += slope_cell(d, q_of("sm", (cell, c, s))) if d else "<td class=muted>–</td>"
        smt.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td>{tds}</tr>")
    smt.append("</table></div>")
    out += smt
    hits = sorted(((k, d) for k, d in cmp_.items() if sig(d, q_of("cmp", k))), key=lambda kd: -abs(kd[1]["b"]))
    out.append("<h3>d. Same trait and direction, different carrier (significant after BH)</h3>"
               + ("<ul>" + "".join(f"<li>{html.escape(cell_name(k[0]))} · {k[1]} pushed {'up' if k[2] > 0 else 'down'} · "
                                   f"{k[3]}: {d['b']:+.0f} points{stars(q_of('cmp', k))}</li>" for k, d in hits[:20])
                  + "</ul>" if hits else "<p class=muted>none</p>"))

    # e. the same persona on opposite roles: reward slopes (as before), per game
    prs = paired(rows, base, lambda r: is_vector(r) and r["condition"] in ("evil_one", "good_one"))
    byp = _by(prs, lambda r: (r["cell"], r["trait"], r["condition"]))
    slope_cols = ["deceiver steered (evil_one)", "honest player steered (good_one)", "difference"]
    mats = []
    for cell in sorted({r["cell"] for r in prs}):
        def get(t, col, cell=cell):
            e, g = _diff_slope(byp.get((cell, t, "evil_one"), []), scale), _diff_slope(byp.get((cell, t, "good_one"), []), scale)
            if col != "difference":
                return e if col == slope_cols[0] else g
            se = math.hypot(e["se"], g["se"])
            b = e["b"] - g["b"]
            p = math.erfc(abs(b / se) / math.sqrt(2)) if se and math.isfinite(se) and se > 0 else float("nan")
            return {"b": b, "se": se, "p": p, "n": e["n"] + g["n"]}
        mats.append(matrix(cell_name(cell), "trait", traits, slope_cols, get)[0])
    out.append("<details><summary>e. The steered agent's own reward per unit c, deceiver vs honest player (SD units)"
               "</summary>" + "".join(mats) + "</details>")

    # findings: what holds across games
    b0s = {c: rate(by.get((c, "none"), [])) for c in cells_}
    high = [cell_name(c) for c, b in b0s.items() if b[2] and b[0] >= 80]
    low = [cell_name(c) for c, b in b0s.items() if b[2] and b[0] <= 20]
    few = [cell_name(c) for c, b in b0s.items() if b[2] < 20]
    f = [f"Unsteered, the deceivers win: {'; '.join(base_txt)}."
         + (f" {', '.join(high)}: near a ceiling, so steering there can mostly only make the deceivers lose." if high
            else "")
         + (f" {', '.join(low)}: the deceivers almost never win, so steering there can mostly only help them." if low
            else "")
         + (" The other games are balanced enough for effects either way." if high or low else "")
         + (f" Fewer than 20 unsteered episodes in {', '.join(few)}: there, tables b–d compare steered runs with each "
            "other rather than with them." if few else "")]
    broke = []
    for cell in cells_:
        for s in (-1, 1):
            d = sm.get((cell, "evil_team", s))
            if d and sig(d, q_of("sm", (cell, "evil_team", s))) and d["b"] < 0:
                broke.append(f"{cell_name(cell)} pushed {'up' if s > 0 else 'down'} ({d['b']:+.0f} points)")
    if broke:
        f.append("<b>Steering the whole deceiving side hard costs it games</b>: strong vs mild steering lowers the "
                 f"deceivers' win rate in {'; '.join(broke)}. Steering one member does not do this as clearly (table c).")
    agree = []
    for c in ("evil_team", "evil_one", "good_team", "good_one", "all"):
        ts = [t for t in traits if any(sig(ud[k], q_of("ud", k)) for k in ud if k[1] == c and k[2] == t)]
        same = [t for t in ts if len({math.copysign(1, ud[(cell, c, t)]["b"]) for cell in cells_
                                      if (cell, c, t) in ud and math.isfinite(ud[(cell, c, t)]["b"])}) == 1]
        if ts:
            agree.append(f"{c}: {len(same)} of {len(ts)} ({', '.join(same) or '–'})")
    flips = []  # the same trait, the whole deceiving side steered: significant in opposite directions in two games
    for t in traits:
        signs = {cell_name(cell): ud[(cell, "evil_team", t)]["b"] for cell in cells_ if (cell, "evil_team", t) in ud
                 and sig(ud[(cell, "evil_team", t)], q_of("ud", (cell, "evil_team", t)))}
        if signs and max(signs.values()) > 0 > min(signs.values()):
            flips.append(f"{t} (" + ", ".join(f"{g} {v:+.0f}" for g, v in signs.items()) + " points)")
    if flips:
        f.append("<b>Flips between games:</b> with the whole deceiving side steered, pushing the trait up rather "
                 f"than down helps the deceivers in one game and hurts them in another for {'; '.join(flips)}: a "
                 "trait's effect on who wins depends on the game's rules, not only on the persona.")
    if agree:
        f.append("<b>Across games</b>, of the traits whose up-vs-down difference is significant in some game, the "
                 f"direction is the same in every game for: {'; '.join(agree)}.")
    sig_ud = sorted(((k, d) for k, d in ud.items() if sig(d, q_of("ud", k))), key=lambda kd: -abs(kd[1]["b"]))
    if sig_ud:
        f.append("Largest up-vs-down differences: " + "; ".join(
            f"{cell_name(k[0])} {k[1]} × {k[2]}: {d['b']:+.0f} points" for k, d in sig_ud[:8]) + ".")
    # the same trait on the whole deceiving vs the whole honest side: does it help whichever side carries it?
    mirror = defaultdict(list)
    for cell in cells_:
        for t in traits:
            e, g = ud.get((cell, "evil_team", t)), ud.get((cell, "good_team", t))
            if e and g and sig(e, q_of("ud", (cell, "evil_team", t))) and sig(g, q_of("ud", (cell, "good_team", t))):
                kind = {(1, -1): "helps whichever side carries it", (-1, 1): "hurts whichever side carries it",
                        (1, 1): "favours the deceivers whoever carries it",
                        (-1, -1): "favours the honest side whoever carries it"}[(int(math.copysign(1, e["b"])),
                                                                                 int(math.copysign(1, g["b"])))]
                mirror[kind].append(f"{cell_name(cell)} × {t}")
    if any(c == "good_team" for _, c in cols):
        f.append("<b>Mirror (evil_team vs good_team, trait pushed up vs down, both significant):</b> "
                 + ("; ".join(f"{kind}: {', '.join(v)}" for kind, v in mirror.items()) if mirror else
                    "no game × trait is significant for both sides yet")
                 + ". 'Helps whichever side carries it' points to a persona that makes a team play better; "
                 "'favours one side whoever carries it' to a trait that matters through the game's roles.")
    tested = lambda kind: sum(1 for k, d in cmp_.items() if k[3] in kind and math.isfinite(d["p"]))
    down_loses = sum(1 for k, d in hits if k[2] < 0 and d["b"] < 0)
    n_team = sum(1 for k, d in hits if k[3] in ("team − one deceiver", "team − one honest"))
    n_role = sum(1 for k, d in hits if k[3] == "one deceiver − one honest")
    t_team, t_role = tested(("team − one deceiver", "team − one honest")), tested(("one deceiver − one honest",))
    f.append(f"<b>Carrier:</b> with the same trait and direction, the whole side vs one member differs significantly "
             f"in {n_team} of {t_team} game × trait × direction cases, one deceiver vs one honest player in {n_role} of "
             f"{t_role} (list d). Most comparisons show no difference at this sample size"
             + (f"; of the differences, {down_loses} of {len(hits)} are the larger or deceiving carrier losing more when "
                "the trait is pushed down." if hits else "."))
    take = [f"<b>Hidden-role games: a whole deceiving side steered hard loses.</b> "
            + (f"Strong rather than mild steering of every deceiver costs them games ({'; '.join(broke[:3])}); "
               if broke else "")
            + f"one deceiver or one honest player steered rarely changes who wins ({n_role} of {t_role} carrier "
            f"comparisons differ); a trait's direction agrees across games for "
            f"{'; '.join(agree[:1]) or 'no trait yet'}"
            + (f"; it flips between games for {', '.join(x.split(' (')[0] for x in flips)}" if flips else "")
            + " (section 22)."]
    return take, section(22, "Hidden-role games: who carries the persona, and what it does to who wins",
                         "In Avalon, Mafia and Werewolf, does it matter whether the whole deceiving side carries the "
                         "persona (evil_team), one deceiver (evil_one), the whole honest side (good_team), one honest "
                         "player (good_one) or everyone (all)? "
                         "Does the same pattern hold across the games?",
                         "Per episode, whether the deceiving side won (spies, mafia, werewolves; 'evil' is the game's "
                         "name for that side, not the evil trait). <b>a</b>: win rate at each c per setting, pooled over "
                         "the 8 traits, the prompt version of evil_team as squares and the unsteered rate at c = 0. "
                         "<b>b</b>: win rate with the trait pushed up (all c &gt; 0) minus pushed down (all c &lt; 0), "
                         "per trait. <b>c</b>: win rate at strong (|c| &gt; 1) minus mild (|c| ≤ 1) steering, per side. "
                         "<b>d</b>: the same trait and direction with a different carrier. <b>e</b>: the steered agent's "
                         "own reward slope as deceiver vs honest player. Tests: two proportions over episodes, "
                         "Benjamini–Hochberg over b, c and d together.",
                         "Positive = the deceivers win more. In b, a cell is a trait's net effect on who wins; read across "
                         "a row to see whether it holds in every game. In c, negative on both sides = the side does "
                         "worse the harder it is steered, whatever the direction (the curves in a bend down at both "
                         "ends). Where the deceivers win nearly every unsteered game (first finding), effects can mostly only be "
                         "losses.",
                         f, "".join(out))


def natural_section(rows, traits, steered, scale):
    """23: do agents that are naturally more X fare like agents steered toward X?"""
    by_cell = _by(rows, lambda r: r["cell"])
    own = _by([r for r in steered if r["condition"] == "one" and _single_vector(r)], lambda r: (r["cell"], r["trait"]))
    res = []
    for cell, cr in sorted(by_cell.items()):
        b0 = [r for r in cr if r["condition"] == "none" and r["reward"] is not None]
        sd = scale.get(cell, 1.0)
        for t in traits:
            u = [r for r in b0 if t in r["mon"]]
            g = [r for r in own.get((cell, t), []) if t in r["mon"]]
            if len(u) < 30 or len(g) < 50:
                continue
            obs = slope([r["mon"][t] for r in u], [r["reward"] / sd for r in u], [r["cluster"] for r in u])
            rew = _diff_slope(g, scale)
            z = slope([r["coef"] for r in g] + [0.0] * len(u), [r["mon"][t] for r in g + u], [r["cluster"] for r in g + u])
            if math.isfinite(obs["b"]) and math.isfinite(rew["b"]) and z["b"] and math.isfinite(z["b"]):
                res.append((cell, t, obs, rew, rew["b"] / z["b"]))
    if not res:
        return [], section(23, "Natural vs steered: does the correlation predict the intervention?", "", "", "", [], "")
    q = bh([x[2]["p"] for x in res] + [x[3]["p"] for x in res])
    qo, qr = q[:len(res)], q[len(res):]
    agree = float(np.mean([np.sign(x[2]["b"]) == np.sign(x[4]) for x in res]))
    both = [(x, qo[i], qr[i]) for i, x in enumerate(res) if sig(x[2], qo[i]) and sig(x[3], qr[i])]
    rc = _rank_corr([x[2]["b"] for x in res], [x[4] for x in res])
    med_o = statistics.median(abs(x[2]["b"]) for x in res)
    med_c = statistics.median(abs(x[4]) for x in res)
    n_obs = sum(1 for i, x in enumerate(res) if sig(x[2], qo[i]))
    opp = [x for x, _, _ in both if np.sign(x[2]["b"]) != np.sign(x[4])]
    pts = [(x[2]["b"], x[4], f"{cell_name(x[0])} × {x[1]}" + ("+" if np.sign(x[2]["b"]) == np.sign(x[4]) else ""))
           for x in res]
    tab = ["<div class=charts><div class=chart><h3>Each point: a game × trait</h3>"
           + scatter_chart(pts, "natural: reward per SD of readout", "steered: reward per SD of readout") + "</div></div>"
           "<div class=wrap><table><tr><th class=l>game</th><th>trait</th><th>natural (unsteered agents)</th>"
           "<th>steered (reward per unit c)</th><th>steered, per SD of readout moved</th></tr>"]
    for i, x in enumerate(res):
        if sig(x[2], qo[i]) or sig(x[3], qr[i]):
            tab.append(f"<tr><td class=l>{html.escape(cell_name(x[0]))}</td><td>{x[1]}</td>{slope_cell(x[2], qo[i])}"
                       f"{slope_cell(x[3], qr[i])}<td>{x[4]:+.3f}</td></tr>")
    tab.append("</table></div>")
    f = [f"{len(res)} game × trait cells. Among unsteered agents the trait's readout goes with reward significantly in "
         f"{n_obs}; the steered effect is significant in {sum(1 for i, x in enumerate(res) if sig(x[3], qr[i]))}.",
         f"The two agree in sign in {100 * agree:.0f}% of cells (50% = chance); rank correlation {rc:+.2f}. Per SD of "
         f"readout, the natural association is {med_o:.3f} SD of reward (median), the steered effect {med_c:.3f}: "
         f"{med_o / med_c if med_c else float('nan'):.0f}× smaller.",
         f"Both significant in {len(both)} cell{'s' if len(both) != 1 else ''}, with opposite signs in {len(opp)}"
         + (": " + "; ".join(f"{cell_name(x[0])} × {x[1]} (natural {x[2]['b']:+.2f}, steered {x[4]:+.3f})" for x in opp[:5])
            if opp else "") + "."]
    take = [f"<b>Correlation is not causation here.</b> Agents that naturally sound more like a trait do not fare like agents "
            f"steered toward it: the signs agree in {100 * agree:.0f}% of game × trait cells (chance 50%) and the natural "
            f"association is about {med_o / med_c if med_c else float('nan'):.0f}× larger per SD (section 23). The game "
            "situation shapes both how an agent talks and whether it wins; only the intervention shows what the persona "
            "itself does."]
    return take, section(23, "Natural vs steered: does the correlation predict the intervention?",
                         "Without steering, one could correlate how agents talk with how they fare. Do agents that are "
                         "naturally more X do as well as agents steered toward X?",
                         "Natural: among unsteered agents of a game, the slope of reward (SD units) on the trait's "
                         "readout (SD units). Steered: the reward effect of the one-agent persona vector per unit c "
                         "(as in section 9), divided by how far that moves the readout per unit c (section 2), giving "
                         "reward per SD of readout. Both per game × trait; BH across all cells.",
                         "If the readout captured a cause of winning, both would agree in sign and size (points on the "
                         "diagonal). Points off the diagonal, or in the other-sign quadrants, are correlations the "
                         "intervention does not reproduce: confounded by the game's course (a losing agent talks "
                         "differently; in hidden-role games the role itself sets both how an agent talks and whether "
                         "it wins). Table: cells where either side is significant.", f, "".join(tab))


def risk_tempo_section(rows, traits, steered, scale):
    """24: does steering make outcomes more variable; do steered games run longer or shorter?"""
    sv = [r for r in steered if _single_vector(r) and r["coef"] is not None]
    by_c = _by(sv, lambda r: abs(r["coef"]))
    tab = ["<h3>Spread of the steered agent's reward change (SD units), by |c|</h3><div class=wrap><table><tr>"
           "<th class=l>|c|</th>" + "".join(f"<th>{c:g}</th>" for c in sorted(by_c)) + "</tr><tr><td class=l>SD</td>"]
    sds = {}
    for c, rs in sorted(by_c.items()):
        v = [r["diff"] / scale.get(r["cell"], 1.0) for r in rs]
        sds[c] = float(np.std(v)) if len(v) > 1 else float("nan")
        tab.append(f"<td title='n={len(v)}'>{sds[c]:.2f}</td>")
    tab.append("</tr></table></div>" + games_note({r["cell"] for r in sv}))
    cell_sd = _by(sv, lambda r: (r["cell"], r["trait"]))
    lo_c, hi_c = (min(by_c), max(by_c)) if by_c else (None, None)
    grow = tot = 0
    for rs in cell_sd.values():
        a = [r["diff"] for r in rs if abs(r["coef"]) == lo_c]
        b = [r["diff"] for r in rs if abs(r["coef"]) == hi_c]
        if len(a) > 10 and len(b) > 10:
            tot += 1
            grow += np.std(b) > np.std(a)
    eps = {r["cluster"]: r for r in rows if (r["condition"] == "all" and r["via"] == "vector" and not r["random"]
                                            and r["team_cond"] is None) or r["condition"] == "none"}
    eps = [r for r in eps.values() if r.get("episode_turns")]
    base_tr = [r["truncated"] for r in eps if r["condition"] == "none"]
    tab.append("<h3>Game length when every agent is steered</h3><div class=wrap><table><tr><th class=l>trait</th>"
               "<th>episode length change per unit c (everyone steered)</th><th>episodes cut off at the turn limit, "
               "|c| = " + f"{hi_c:g}" + "</th></tr>")
    fast = []
    for t in traits:
        pts = [r for r in eps if r["trait"] == t or r["condition"] == "none"]
        s = slope([r["coef"] if r["condition"] != "none" else 0.0 for r in pts],
                  [math.log(r["episode_turns"]) for r in pts], [r["cluster"] for r in pts], [r["cell"] for r in pts])
        tr = [r["truncated"] for r in pts if r["coef"] is not None and abs(r["coef"]) == hi_c]
        pct = 100 * (math.exp(s["b"]) - 1) if math.isfinite(s["b"]) else float("nan")
        if sig(s) and abs(pct) >= 3:
            fast.append(f"{t} ({pct:+.1f}%)")
        tab.append(f"<tr><td class=l>{t}</td><td title='SE {fmt(s['se'])} (log scale), p {fmt(s['p'], 4)}'>{pct:+.1f}%"
                   f"<span class=sig>{stars(s['p'])}</span></td><td>{100 * np.mean(tr) if tr else float('nan'):.1f}%</td></tr>")
    tab.append("</table></div>" + games_note({r["cell"] for r in eps if r["condition"] != "none"}))
    ratio = sds[hi_c] / sds[lo_c] if lo_c in sds and sds[lo_c] else float("nan")
    f = [f"Spread of the reward change: SD {sds.get(lo_c, float('nan')):.2f} at |c| = {lo_c:g}, {sds.get(hi_c, float('nan')):.2f} "
         f"at |c| = {hi_c:g} ({ratio:.2f}×); it grows from the weakest to the strongest coefficient in {grow} of {tot} "
         "game × trait cells. Steering shifts outcomes; it does not make them more erratic."
         if math.isfinite(ratio) and ratio < 1.15 else
         f"Spread of the reward change grows with strength: SD {sds.get(lo_c, float('nan')):.2f} → {sds.get(hi_c, float('nan')):.2f} "
         f"({ratio:.2f}×), in {grow} of {tot} cells.",
         ("Games change length by ≥ 3% per unit c when everyone is steered for: " + "; ".join(fast) + "." if fast else
          "Steering everyone changes game length by less than 3% per unit c for every trait.")
         + f" Unsteered games are cut off at the turn limit in {100 * np.mean(base_tr) if base_tr else float('nan'):.1f}% of cases."]
    take = []
    return take, section(24, "Risk and tempo",
                         "Besides shifting the average outcome, does steering make outcomes more variable (riskier)? "
                         "And do steered groups play faster or slower?",
                         "Risk: the standard deviation of the steered agent's reward change (one-agent persona vector, "
                         "SD units, as in section 9) at each |c|, pooled, and per game × trait between the weakest and "
                         "the strongest coefficient. Tempo: per episode with every agent steered, the log of the number "
                         "of turns over the coefficient (game fixed effects; shown as % per unit c), and the share of "
                         "episodes stopped at the turn limit at the strongest coefficient.",
                         "A rising SD = stronger steering makes the agent's results less predictable. Negative length = "
                         "games end sooner (decisions faster, or an early loss).", f, "".join(tab))


# ---------------------------------------------------------------- text vs image

def _tokens_total(rs):
    """Tokens read by all agents of these rows, e.g. '3.2M' ('–' when not counted; '≥' when only some are)."""
    have = [r["tokens_in"] for r in rs if r.get("tokens_in") is not None]
    if not have:
        return "–"
    total = sum(have)
    size = f"{total / 1e9:.2f}B" if total >= 1e9 else f"{total / 1e6:.1f}M" if total >= 1e6 else f"{total / 1e3:.0f}k"
    return size if len(have) == len(rs) else f"≥ {size}"


def per_turn(rs, key):
    """Mean of an agent-episode total (e.g. tokens read) per turn, over the agents that have it."""
    vals = [r[key] / r["turns"] for r in rs if r.get(key) is not None and r["turns"]]
    return float(np.mean(vals)) if vals else float("nan")


def mode_baselines(rows):
    """{group: {"text": stats, "image": stats, "paired": [(cluster, image - text reward)]}} for the games played both
    ways, from their unsteered agents: the same seeds give the same deals and seats in both modes, so each seat's
    reward with images is paired with its reward with text."""
    none = [r for r in rows if r["condition"] == "none"]
    modes = defaultdict(set)
    for r in rows:
        modes[r["group"]].add(r["obs"])
    out = {}
    for g in sorted(k for k, m in modes.items() if {"text", "image"} <= m):
        by = {m: [r for r in none if r["group"] == g and r["obs"] == m] for m in ("text", "image")}
        if not by["text"] or not by["image"]:
            continue
        res = {}
        for m, rs in by.items():
            rw = [r["reward"] for r in rs if r["reward"] is not None]
            res[m] = {"episodes": len({r["episode"] for r in rs}), "reward": float(np.mean(rw)) if rw else float("nan"),
                      "win": 100 * float(np.mean([x > 0 for x in rw])) if rw else float("nan"),
                      "binary": set(np.round(rw, 6)) <= {0.0, 1.0} or set(np.round(rw, 6)) <= {-1.0, 1.0},
                      "unparseable": 100 * sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs)),
                      "chars": float(np.mean([r["chars"] for r in rs if r["chars"]] or [float("nan")])),
                      "tokens_in": per_turn(rs, "tokens_in"), "tokens_out": per_turn(rs, "tokens_out"),
                      "length": float(np.mean([r["episode_turns"] for r in rs if r["episode_turns"]] or [float("nan")]))}
        text = {(r["episode"], r["agent"]): r["reward"] for r in by["text"] if r["reward"] is not None}
        res["paired"] = [((g, r["episode"]), r["reward"] - text[(r["episode"], r["agent"])]) for r in by["image"]
                         if r["reward"] is not None and (r["episode"], r["agent"]) in text]
        out[g] = res
    return out


def mode_curves(rows, steered, scale, traits):
    """Per trait, over c, text vs image (games played both ways, persona vector, no side settings): the steered
    agents' readout of the trait and their paired reward change (SD units). {trait: {"readout": series,
    "reward": series}} with series {mode: [(c, mean, half CI)]}."""
    groups = defaultdict(set)
    for r in rows:
        groups[r["group"]].add(r["obs"])
    both = {g for g, m in groups.items() if {"text", "image"} <= m}
    out = {}
    for t in traits:
        rd = [r for r in rows if is_vector(r, t) and r["team_cond"] is None and r["group"] in both and t in r["mon"]]
        rw = [r for r in steered if r["trait"] == t and r["group"] in both and math.isfinite(r["diff"])]
        out[t] = {kind: {m: [(c, *mean_ci([v(r) for r in rs], [r["cluster"] for r in rs])[:2])
                             for c, rs in sorted(_by([r for r in src if r["obs"] == m and r["coef"]],
                                                     lambda r: r["coef"]).items())] + [(0.0, 0.0, float("nan"))]
                         for m in ("text", "image")}
                  for kind, src, v in (("readout", rd, lambda r: r["mon"][t]),
                                       ("reward", rw, lambda r: r["diff"] / scale.get(r["cell"], 1.0)))}
    return out


def modes_table(base):
    """Unsteered play, text vs image, per game (html)."""
    out = ["<h3>Without steering: how image play differs from text play</h3><div class=wrap><table><tr>"
           "<th class=l>game</th><th>result: text</th><th>result: image</th><th>image − text (same seats, seeds)</th>"
           "<th>unparseable: text / image</th><th>answer length: text / image</th>"
           "<th>tokens read per turn: text / image</th><th>tokens generated per turn: text / image</th>"
           "<th>turns per episode: text / image</th></tr>"]
    show = lambda d: f"{d['win']:.0f}% wins" if d["binary"] else fmt(d["reward"], 2)
    two = lambda a, b, d=0: f"{fmt(a, d)} / {fmt(b, d)}" if math.isfinite(a) or math.isfinite(b) else "–"
    for g, d in base.items():
        t, i = d["text"], d["image"]
        m, half, n = mean_ci([x for _, x in d["paired"]], [k for k, _ in d["paired"]])
        k = 100 if t["binary"] else 1
        diff = (f"{k * m:+.{0 if t['binary'] else 2}f}{' points' if t['binary'] else ''} ± {k * half:.{0 if t['binary'] else 2}f}"
                if math.isfinite(half) else "–")
        out.append(f"<tr><td class=l>{html.escape(g)}</td><td>{show(t)} <span class=muted>(n={t['episodes']})</span></td>"
                   f"<td>{show(i)} <span class=muted>(n={i['episodes']})</span></td><td>{diff}</td>"
                   f"<td>{t['unparseable']:.1f}% / {i['unparseable']:.1f}%</td><td>{two(t['chars'], i['chars'])}</td>"
                   f"<td>{two(t['tokens_in'], i['tokens_in'])}</td><td>{two(t['tokens_out'], i['tokens_out'])}</td>"
                   f"<td>{two(t['length'], i['length'], 1)}</td></tr>")
    out.append("</table></div><p class=muted>Unsteered agents only. Results are the game's own (win share, or the mean "
               "score); image − text pairs each seat with itself in the same deal (95% CI). Tokens read: every "
               "attempt's prompt, text and image tokens; '–' where the counts are not in the data yet "
               "(scripts/backfill_tokens.py).</p>")
    return "".join(out)


# ---------------------------------------------------------------- per-setting breakdowns under every section

ALREADY_BY_SETTING = {"7": "rows = settings", "8": "rows = settings", "16": "per game: one chart per setting",
                      "19": "vector / prompt / both, and table h for everyone and the deceiving side",
                      "21": "one vs all", "22": "evil_team, evil_one, good_team, good_one, all, one"}


SETTING_GAMES = {}  # setting -> the games it was played in (set by setting_pages), for the by-setting tables' notes


def setting_table(title, sets, traits, get, read=""):
    """A collapsible table: one row per setting, one column per trait. get(setting, trait) -> a slope dict (stars:
    Benjamini-Hochberg over the table), a str, or None."""
    found = {(c, t): get(c, t) for c in sets for t in traits}
    keys = [k for k, v in found.items() if isinstance(v, dict) and math.isfinite(v.get("p", float("nan")))]
    q = dict(zip(keys, bh([found[k]["p"] for k in keys])))
    out = [f"<details class=bysetting><summary>By setting: {html.escape(title)}</summary><div class=wrap><table><tr>"
           "<th class=l>setting</th>" + "".join(f"<th>{html.escape(t)}</th>" for t in traits) + "</tr>"]
    for c in sets:
        tds = []
        for t in traits:
            v = found[(c, t)]
            tds.append(slope_cell(v, q.get((c, t))) if isinstance(v, dict) else
                       f"<td>{v}</td>" if v else "<td class=muted>–</td>")
        out.append(f"<tr><td class=l>{html.escape(c)}</td>{''.join(tds)}</tr>")
    per = defaultdict(set)
    for (c, _), v in found.items():
        if isinstance(v, dict):
            per[c] |= set(v.get("games", ()))
    note = ""
    if not any(per.values()):
        per = {c: SETTING_GAMES.get(c, set()) for c in sets}
    if any(per.values()):
        lists = {c: ", ".join(cell_name(g) for g in sorted(gs)) for c, gs in per.items() if gs}
        note = ("<p class=games><b>Games:</b> " + "; ".join(f"<b>{html.escape(c)}</b> ({len(per[c])}): {html.escape(l)}"
                                                         for c, l in lists.items()) + "</p>"
                if len(set(lists.values())) > 1 else games_note(next(gs for gs in per.values() if gs)))
    out.append("</table></div>" + note + (f"<p class=muted>{read}</p>" if read else "") + "</details>")
    return "".join(out)


def _auc(rs, t):
    """Mean over episodes of P(a steered agent's readout is more extreme, in its push's direction, than an
    unsteered co-player's): 0.5 = the monitor cannot tell them apart."""
    vals = []
    for cl, ep in _by([r for r in rs if t in r["mon"]], lambda r: r["cluster"]).items():
        st, ot = [r for r in ep if r["steered"]], [r["mon"][t] for r in ep if not r["steered"]]
        if st and ot and st[0]["coef"]:
            sign = 1 if st[0]["coef"] > 0 else -1
            vals.append(float(np.mean([1.0 if sign * (s["mon"][t] - o) > 0 else 0.5 if s["mon"][t] == o else 0.0
                                       for s in st for o in ot])))
    return f"{np.mean(vals):.2f}" if vals else None


def setting_pages(rows, traits, base, scale):
    """{section number: html}: each section's measure repeated per setting (who is steered, and how)."""
    by_c = _by(rows, lambda r: r["condition"])
    none = by_c.get("none", [])
    sets = [c for c in SETTING_ORDER if c in by_c]
    SETTING_GAMES.clear()
    SETTING_GAMES.update({c: {r["cell"] for r in rs} for c, rs in by_c.items()})
    sub = lambda c: by_c[c] + none
    steer = lambda r: r["steered"]
    pr = _by(paired(rows, base, lambda r: r["condition"] != "none" and r["trait"] is not None), lambda r: r["condition"])
    pages = {}

    # 1. coverage
    cov = ["<details class=bysetting><summary>By setting: episodes and clean runs</summary><div class=wrap><table><tr>"
           "<th class=l>setting</th><th>episodes</th><th>cut off at the turn limit</th><th>unparseable answers</th>"
           "</tr>"]
    for c in ["none"] + sets:
        rs = by_c.get(c, [])
        eps = {r["cluster"]: r["truncated"] for r in rs}
        cov.append(f"<tr><td class=l>{c}</td><td>{len(eps)}</td><td>{100 * sum(eps.values()) / max(len(eps), 1):.1f}%</td>"
                   f"<td>{100 * sum(r['fallbacks'] for r in rs) / max(1, sum(r['turns'] for r in rs)):.1f}%</td></tr>")
    pages["1"] = "".join(cov) + "</table></div><p class=games><b>Games:</b> " + "; ".join(
        f"<b>{html.escape(c)}</b> ({len(SETTING_GAMES.get(c, ()))})" for c in ["none"] + sets) \
        + " — the games per setting are listed under the other by-setting tables</p></details>"
    # 2. manipulation
    pages["2"] = setting_table("readout of the steered trait, SD per unit c", sets, traits,
                               lambda c, t: readout_slope(sub(c), t, steer),
                               "Prompt settings: the sign of the instruction is the coefficient (±1).")

    # 3. specificity: the strongest other trait each vector moves, relative to its own
    def side(c, t):
        own = readout_slope(sub(c), t, steer)
        if not math.isfinite(own.get("b", float("nan"))) or not own["b"]:
            return None
        other = {u: readout_slope(sub(c), t, steer, readout=u)["b"] for u in traits if u != t}
        other = {u: b for u, b in other.items() if math.isfinite(b)}
        if not other:
            return None
        u = max(other, key=lambda k: abs(other[k]))
        return f"{html.escape(u)} {other[u] / own['b']:+.0%}"
    pages["3"] = setting_table("the other trait that moves most (its slope as % of the steered trait's)", sets,
                               traits, side, "all_random: a random direction carries the trait's name only for its norm, "
                               "so its 'own' slope is near 0 and the % is not meaningful there.")
    # 4. detectability
    pages["4"] = setting_table("AUC: steered agents vs their unsteered co-players (0.5 = invisible)", sets, traits,
                               lambda c, t: _auc([r for r in by_c[c] if r["trait"] == t], t),
                               "Settings where every agent is steered have no unsteered co-player to compare with (–).")
    # 5. amplify vs suppress
    def updown(c, t):
        a = readout_slope(sub(c), t, lambda r: r["steered"] and r["coef"] > 0)
        b = readout_slope(sub(c), t, lambda r: r["steered"] and r["coef"] < 0)
        return f"{fmt(a['b'], 1)} / {fmt(b['b'], 1)}" if math.isfinite(a.get("b", float("nan"))) else None
    pages["5"] = setting_table("readout slope pushing up / pushing down (SD per unit c)", sets, traits, updown)

    # 6. prompt vs vector: the same carriers, given the vector (c = ±1) or the prompt
    pairs = [("one agent", "one", "prompt_one"), ("everyone", "all", "prompt_all"),
             ("deceiving side", "evil_team", "evil_team_prompt")]
    pairs = [p for p in pairs if p[1] in by_c and p[2] in by_c]

    def pv(c, t):
        _, vec, pro = next(p for p in pairs if p[0] == c)
        m = lambda rs: _episode_mean([math.copysign(1, r["coef"]) * r["mon"][t] for r in rs],
                                     [r["cluster"] for r in rs])[0]
        v = m([r for r in by_c[vec] if r["steered"] and r["trait"] == t and abs(r["coef"]) == 1 and t in r["mon"]])
        p = m([r for r in by_c[pro] if r["steered"] and r["trait"] == t and t in r["mon"]])
        return f"{fmt(v, 1)} / {fmt(p, 1)}" if math.isfinite(v) or math.isfinite(p) else None
    if pairs:
        pages["6"] = setting_table("readout the intended way: vector at c = ±1 / prompt (SD)", [p[0] for p in pairs],
                                   traits, pv)
    # 9, 10. own and co-players' reward
    pages["9"] = setting_table("steered agents' reward, SD per unit c", sets, traits,
                               lambda c, t: _diff_slope([r for r in pr.get(c, []) if r["steered"] and r["trait"] == t], scale))
    pages["10"] = setting_table("unsteered co-players' reward, SD per unit c", sets, traits,
                                lambda c, t: _diff_slope([r for r in pr.get(c, []) if not r["steered"] and r["trait"] == t],
                                                         scale))
    # 11. the deceiving side's reward, whoever is steered
    rsets = [c for c in sets if any(r["team"] == "evil" for r in pr.get(c, []))]
    pages["11"] = setting_table("the deceiving side's reward, SD per unit c (whoever is steered)", rsets, traits,
                                lambda c, t: _diff_slope([r for r in pr.get(c, []) if r["team"] == "evil"
                                                          and r["trait"] == t], scale))

    # 11b. readout -> reward among the steered agents
    def assoc(c, t):
        rs = [r for r in pr.get(c, []) if r["steered"] and r["trait"] == t and t in r["mon"]]
        return slope([r["mon"][t] for r in rs], [r["diff"] / scale.get(r["cell"], 1.0) for r in rs],
                     [r["cluster"] for r in rs], [r["cell"] for r in rs]) if len(rs) > 20 else None
    pages["11b"] = setting_table("reward change per +1 SD of readout (steered agents; correlational)", sets, traits, assoc)
    # 12. image - text
    pages["12"] = setting_table("readout slope with images minus with text (SD per unit c)", sets, traits,
                                lambda c, t: obs_compare(sub(c), t, lambda s_, t_: readout_slope(s_, t_, steer))[0]["diff"])

    # 13. unparseable answers
    def broken(c, t):
        rs = [r for r in by_c[c] if r["steered"] and r["trait"] == t and r["turns"]] + [r for r in none if r["turns"]]
        return slope([r["coef"] if r["condition"] != "none" else 0.0 for r in rs],
                     [100 * r["fallbacks"] / r["turns"] for r in rs], [r["cluster"] for r in rs], [r["cell"] for r in rs])
    pages["13"] = setting_table("unparseable answers, percentage points per unit c", sets, traits, broken)
    # 14. over the episode
    pages["14"] = setting_table("late − early readout per unit c (opposite sign to section 2 = it fades)", sets, traits,
                                lambda c, t: phase_slope(sub(c), t, steer,
                                                         lambda r, u: r["phase"]["late"][u] - r["phase"]["early"][u]))

    # 15. actions: the numeric choice, in SD of the game's unsteered agents
    num0 = {cell: (float(np.mean(v)), float(np.std(v)) or 1.0) for cell, v in
            _by([r for r in none if r["act_num"] is not None], lambda r: r["cell"]).items()
            for v in [[r["act_num"] for r in v]] if len(v) > 1}

    def numeric(c, t):
        rs = [r for r in sub(c) if r["act_num"] is not None and r["cell"] in num0
              and (r["condition"] == "none" or (r["steered"] and r["trait"] == t))]
        if not any(r["condition"] != "none" for r in rs):
            return None
        return slope([r["coef"] if r["condition"] != "none" else 0.0 for r in rs],
                     [(r["act_num"] - num0[r["cell"]][0]) / num0[r["cell"]][1] for r in rs],
                     [r["cluster"] for r in rs], [r["cell"] for r in rs])
    pages["15"] = setting_table("the agent's numeric choice (bid, guess, amount), SD per unit c, games that have one",
                                sets, traits, numeric)

    # 17. footprint: where the steered trait ranks among all monitored personas, at the strongest |c|
    def rank(c, t):
        rs = [r for r in by_c[c] if r["steered"] and r["trait"] == t and r["coef"]]
        if not rs:
            return None
        cmax, out = max(abs(r["coef"]) for r in rs), []
        for s in (1, -1):
            fp = fingerprint([r for r in rs if r["coef"] == s * cmax])
            if t in fp:
                out.append(f"{'+' if s > 0 else '−'} {1 + sorted(fp, key=lambda u: -s * fp[u]).index(t)}")
        return " · ".join(out) or None
    pages["17"] = setting_table("rank of the steered trait among all monitored personas (1 = moves most)", sets,
                                traits, rank)

    # 18. naming: centroids learnt on one-agent steering (even episodes) name the trait in every setting
    one = [r for r in rows if _single_vector(r) and r["mon"]]
    if one:
        mon = sorted({u for r in one for u in r["mon"]})
        vec = lambda r: np.nan_to_num(np.array([r["mon"].get(u, 0.0) for u in mon]))
        train = [r for r in one if r["episode"] % 2 == 0]
        cls = sorted({(r["trait"], r["coef"] > 0) for r in train})
        C = np.stack([np.mean([vec(r) for r in train if (r["trait"], r["coef"] > 0) == k], 0) for k in cls])

        def named(c, t):
            rs = [r for r in by_c[c] if r["steered"] and r["trait"] == t and r["mon"] and r["coef"]
                  and not (c == "one" and r["episode"] % 2 == 0)]
            if not rs:
                return None
            X = np.stack([vec(r) for r in rs])
            pred = (-2 * X @ C.T + (C * C).sum(1)).argmin(1)
            return f"{100 * np.mean([cls[j][0] == t for j in pred]):.0f}%"
        pages["18"] = setting_table("named right by the monitor trained on one-agent steering (chance "
                                    f"{100 / len(traits):.0f}%)", sets, traits, named)

    # 20. length held fixed
    def kept(c, t):
        rs = [r for r in sub(c) if r["chars"] and t in r["mon"] and (r["condition"] == "none" or (r["steered"] and r["trait"] == t))]
        if not any(r["condition"] != "none" for r in rs):
            return None
        x = [r["coef"] if r["condition"] != "none" else 0.0 for r in rs]
        y, ll = [r["mon"][t] for r in rs], [math.log(max(r["chars"], 1.0)) for r in rs]
        cl, fe = [r["cluster"] for r in rs], [r["cell"] for r in rs]
        b0, b1 = slope(x, y, cl, fe)["b"], ols(y, list(zip(x, ll)), cl, fe)[0]["b"]
        return f"{100 * b1 / b0:.0f}%" if b0 and math.isfinite(b0) and math.isfinite(b1) else None
    pages["20"] = setting_table("share of the readout shift left with answer length held fixed", sets, traits, kept)

    # 23. steered reward per SD of readout (compare with section 23's natural association)
    def per_sd(c, t):
        rew = _diff_slope([r for r in pr.get(c, []) if r["steered"] and r["trait"] == t], scale)["b"]
        rd = readout_slope(sub(c), t, steer)["b"]
        return f"{rew / rd:+.3f}" if rd and math.isfinite(rd) and math.isfinite(rew) else None
    pages["23"] = setting_table("steered effect: reward (SD) per SD of readout", sets, traits, per_sd)

    # 24. risk: spread of the reward change at mild and strong steering
    def spread(c, t):
        d = lambda keep: [r["diff"] / scale.get(r["cell"], 1.0) for r in pr.get(c, [])
                          if r["steered"] and r["trait"] == t and r["coef"] and keep(abs(r["coef"]))
                          and math.isfinite(r["diff"])]
        lo_, hi_ = d(lambda a: a <= 0.5 or a == 1 and c.startswith("prompt")), d(lambda a: a >= 2)
        return f"{np.std(lo_):.2f} → {np.std(hi_):.2f}" if len(lo_) > 5 and len(hi_) > 5 else None
    pages["24"] = setting_table("SD of the steered agents' reward change: |c| = 0.5 → |c| = 2", sets, traits, spread)
    for num, what in ALREADY_BY_SETTING.items():
        pages[num] = f"<p class=muted>By setting: this section is already split by setting ({what}).</p>"
    return pages


def monitor_how(n_mon, layers) -> str:
    """How the monitor (readout) works, step by step (html list)."""
    at = f"block {', '.join(str(int(m)) for m in layers)}" if layers else "the monitor layer"
    return ("<ol style='margin:0;padding-left:18px'>"
            "<li><b>A second, unsteered copy of the model reads every turn.</b> The same model, on its own GPU, is "
            "given the agent's exact prompt and answer (image turns with their images) as a prefill, as if it had "
            "written them itself; it generates nothing new.</li>"
            "<li><b>It records the hidden states</b> (the residual stream after each decoder block) of every token, "
            "with vLLM-Hook.</li>"
            "<li><b>Only the answer counts:</b> the prompt's tokens are dropped and the answer's hidden states "
            "averaged, one vector per turn.</li>"
            f"<li><b>Projected on each persona:</b> that vector's dot product with the unit-length PersVecFast vector "
            f"of each of the {n_mon} monitored personas, one number per persona: how strongly the model represents "
            "that persona in what the agent wrote.</li>"
            f"<li><b>Per agent and episode:</b> the mean over its turns at {at} (and per third of its turns, for "
            "section 14).</li>"
            "<li><b>Scaled</b> to the unsteered agents of the same game: 0 = a typical unsteered agent, +1 = one "
            "standard deviation more of the persona.</li></ol>"
            "Because the reader is unsteered, it measures what the text conveys, not the injected vector, which is why "
            "it can read prompted agents too. Its limits: the persona vectors overlap, so one number never means one "
            "trait alone (sections 3, 17), and answer length and style leak into it (section 20).")


def build(rows, sweep=None, logs_dir=None, norms=None) -> str:
    rows = zscored(rows)
    traits = sorted({r["trait"] for r in rows if r["trait"]})
    cells = sorted({r["cell"] for r in rows})
    CELL_NAMES.clear()
    CELL_NAMES.update(cell_name(c) for c in cells)
    base = baselines(rows)
    scale = reward_scale(rows)
    key, body = [], []  # key findings (top of the page) and the sections

    # 1. coverage
    tab = ["<div class=wrap><table><tr><th class=l>game</th><th>runs</th><th>episodes</th><th>unsteered episodes</th>"
           "<th>cut off at the turn limit</th><th>unparseable answers</th><th>tokens read per turn</th>"
           "<th>tokens generated per turn</th><th>tokens read, all agents</th></tr>"]
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell]
        eps = {r["cluster"]: r["truncated"] for r in cr}
        turns, fb = sum(r["turns"] for r in cr), sum(r["fallbacks"] for r in cr)
        tab.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td><td>{len({r['run_id'] for r in cr})}</td>"
                   f"<td>{len(eps)}</td><td>{len({r['cluster'] for r in cr if r['condition'] == 'none'})}</td>"
                   f"<td>{100 * sum(eps.values()) / max(len(eps), 1):.0f}%</td><td>{100 * fb / max(turns, 1):.1f}%</td>"
                   f"<td>{fmt(per_turn(cr, 'tokens_in'), 0) or '–'}</td><td>{fmt(per_turn(cr, 'tokens_out'), 0) or '–'}</td>"
                   f"<td>{_tokens_total(cr)}</td></tr>")
    tab.append("</table></div>")
    bad = [cell_name(c) for c in cells
           if sum(r["fallbacks"] for r in rows if r["cell"] == c) > 0.05 * max(1, sum(r["turns"] for r in rows if r["cell"] == c))]
    body.append(section(1, "Coverage", "How much data is there per game, and did the games run cleanly?",
                        "Finished episodes per game; the share of episodes stopped at the turn limit; the share of "
                        "agent turns whose answer could not be parsed even after the retries (the agent then plays a "
                        "default move).", "Unsteered episodes are the reference for everything else: few of them make "
                        "all later numbers noisy. Unparseable answers above a few % point to a prompt/parser problem.",
                        [f"{len({r['cluster'] for r in rows})} episodes over {len(cells)} games."
                         + (f" The agents read {_tokens_total(rows)} tokens (prompt text and images, every attempt; "
                            f"{fmt(per_turn(rows, 'tokens_in'), 0)} per turn) and generated "
                            f"{_tokens_total([{**r, 'tokens_in': r['tokens_out']} for r in rows])}"
                            "; the monitor reads each turn's prompt and answer once more."
                            if any(r.get("tokens_in") is not None for r in rows) else
                            " Token counts are not in this data yet (scripts/backfill_tokens.py adds them).")]
                        + ([f"More than 5% unparseable answers in: {', '.join(bad)}."] if bad else []), "".join(tab)))

    # 2. manipulation
    man = manipulation(rows, traits)
    q_man = bh([man[t]["vector"]["p"] for t in traits] + [man[t]["random"]["p"] for t in traits])
    tab = ["<div class=wrap><table><tr><th class=l>trait</th><th>persona vector: readout per unit c (SD)</th>"
           "<th>random direction (control): readout per unit c (SD)</th><th>prompt, + instruction: mean readout (SD)"
           "</th><th>prompt, − instruction: mean readout (SD)</th></tr>"]
    for i, t in enumerate(traits):
        pp, pn = man[t]["prompt"][1], man[t]["prompt"][-1]
        tab.append(f"<tr><td class=l>{t}</td>{slope_cell(man[t]['vector'], q_man[i])}"
                   f"{slope_cell(man[t]['random'], q_man[len(traits) + i])}"
                   f"<td>{fmt(pp[0], 2)} ± {fmt(pp[1], 2)}</td><td>{fmt(pn[0], 2)} ± {fmt(pn[1], 2)}</td></tr>")
    tab.append("</table></div>" + games_note({r["cell"] for r in rows if r["steered"] and r["trait"]}) + LEGEND
               + "<div class=charts>")
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
    pos = sum(1 for k, s in pg_found.items() if sig(s, pg_q[k]) and s["b"] > 0)
    n_pg = sum(1 for s in pg_found.values() if s and math.isfinite(s.get("b", float("nan"))))
    extra = [f"<b>The steering works in nearly every game.</b> The steered trait moves the intended way in {pos} of "
             f"{n_pg} game × trait cells" + (f", the opposite way in {neg}" if neg else "") + " (section 2): the "
             "language effect does not depend on which game is played."] if n_pg else []
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
                 + ("the persona vectors are strongly entangled (they share a common 'persona' axis), so one "
                    "trait's readout alone does not tell which trait was steered (the whole profile partly does, "
                    "section 18)." if entangled > len(traits) * 2 else
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
    tab = ["<div class=wrap><table><tr><th class=l>trait \\ steering strength</th>"
           + "".join(f"<th>AUC at c = {c:g}</th>" for c in coefs) + "</tr>"]
    for t in traits:
        tds = []
        for c in coefs:
            m, h, n = det[t].get(c, (float("nan"), float("nan"), 0))
            shade = (f' style="background:rgba(192,57,43,{max(0.0, min(1.0, (m - 0.5) * 2)) * 0.55:.2f})"'
                     if math.isfinite(m) else "")
            tds.append(f'<td{shade} title="episodes={n}">{fmt(m, 2)}{"" if not math.isfinite(h) else f" ±{h:.2f}"}</td>')
        tab.append(f"<tr><td class=l>{t}</td>{''.join(tds)}</tr>")
    tab.append("</table></div>" + games_note({r["cell"] for r in rows if r["select"] == "one" and r["steered"]
                                              and r["via"] in STEERED_VIA and not r["random"] and r["team_cond"] is None}))
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
    n_mon = len({t for r in rows for t in r["mon"]})
    ch, agree, corr = channels(rows, traits, base, scale)
    sides = [(t, s) for t in traits if t in ch for s in (1, -1) if s in ch[t]]
    sgn = lambda s: "+" if s > 0 else "−"

    def eq_text(eq):
        x, note = eq
        return "–" if not math.isfinite(x) else f"beyond {x:+g}" if note == ">" else "≤ 0" if note == "<" else f"{x:+.1f}"

    g6 = games_note({r["cell"] for r in rows if r["via"] == "prompt" or r["condition"] == "both_one"},
                    "Games with a prompt arm, in tables a–d")
    # a. strength
    tab_a = ["<h3>a. Strength: how far does each move the trait's own readout?</h3><div class=wrap><table><tr>"
             "<th class=l>trait</th><th>prompt, + instruction: readout (SD)</th><th>vector strength c with the same "
             "readout</th><th>vector at max + c: readout (SD)</th><th>prompt, − instruction: readout (SD)</th>"
             "<th>vector strength c with the same readout</th><th>vector at max − c: readout (SD)</th>"
             "<th>vector alone: readout per unit c</th><th>prompt + vector: readout per unit c</th></tr>"]
    for t in traits:
        if t not in ch:
            continue
        both = readout_slope(rows, t, lambda r: r["steered"] and r["via"] == "both" and not r["random"])
        only = readout_slope(rows, t, lambda r: r["steered"] and r["via"] == "vector" and not r["random"])
        cols = "".join(f"{mean_cell(ch[t][s]['shift'])}<td>{eq_text(ch[t][s]['eq'])}</td><td>{fmt(ch[t][s]['vmax'], 2)}</td>"
                       if s in ch[t] else "<td class=muted>–</td>" * 3 for s in (1, -1))
        tab_a.append(f"<tr><td class=l>{t}</td>{cols}{slope_cell(only)}{slope_cell(both)}</tr>")
    tab_a.append("</table></div>" + g6)

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
    tab_b.append("</table></div>" + g6)

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
    tab_c.append("</table></div>" + g6)
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
    tab_d.append("</table></div>" + g6)

    if auc_p:
        extra.append(f"<b>A persona in the prompt is no harder to spot.</b> The monitor singles out a prompted agent "
                     f"with mean AUC {np.mean(auc_p):.2f}, a vector-steered one (strongest c) with {np.mean(auc_v):.2f} "
                     "(section 6): putting the persona in the system prompt does not hide it from an activation monitor.")
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
            f6.append(f"<b>Direction:</b> the prompt and the vector move the {n_mon} monitored personas in a similar pattern "
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
                        "readout (interpolated on the vector's dose-response curve). <b>b</b>: the shift of all "
                        f"{n_mon} "
                        "monitored traits as a profile; cosine 1 = same pattern, 0 = unrelated. <b>c</b>: paired reward "
                        "(same seat, same seed, unsteered; SD units) of the one steered agent (prompt_one vs one), of "
                        "the deceiving side when all its members are steered (evil_team_prompt vs evil_team), and the "
                        "group's total when every agent is steered (prompt_all vs all; in hidden-role games this "
                        "total only says which side won, the honest side being bigger; "
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
    roles = role_cells(rows)
    wel = welfare([r for r in rows if r["condition"] != "none" and r["cell"] not in roles],
                  [r for r in rows if r["condition"] == "none" and r["cell"] not in roles])
    w7, w_found, w_q = matrix("Group welfare: everyone's total reward, games without hidden roles (SD per unit c)",
                              "condition", [c for c in conds if not c.startswith(("evil_", "good_"))], traits,
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
                        "off. Welfare leaves out the hidden-role games: there the total only says which side won "
                        "(the honest side is bigger, so its win adds more), which section 22 measures directly.",
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
        f11 = [("Steering the whole deceiving side changes its reward for: " + "; ".join(th)
                + " (SD per unit c; the reward is winning, so positive = the deceivers win more when pushed toward "
                  "the trait; win rates per game in section 22).") if th else
               "No significant effect on the deceiving side's reward."]
        key.append(f11[0])
        body.append(section(11, "Steering the deceiving side (hidden-role games)",
                            "If the whole evil team (spies, werewolves, mafia) is steered, does it win more or less?",
                            "The evil agents' reward vs the same seats in the unsteered episode with the same seed, per "
                            "unit coefficient.", "One side wins: the deceivers' gain is the honest side's loss. Where "
                            "the deceivers already win nearly every unsteered game, steering can mostly only make them "
                            "lose; section 22 gives the unsteered win rate of every game.",
                            f11, p11 + g11))

    # 11b. link
    lk = {t: link(steered, t, scale) for t in traits}
    q_lk = bh([lk[t]["total"]["p"] for t in traits] + [lk[t]["assoc"]["p"] for t in traits]
              + [lk[t]["direct"]["p"] for t in traits])
    n_t = len(traits)
    tab = ["<div class=wrap><table><tr><th class=l>trait</th><th>steering strength → reward (SD per unit c)</th>"
           "<th>readout → reward (SD per SD of readout)</th><th>steering strength → reward, readout held fixed</th>"
           "<th>share of the effect that goes with the readout</th></tr>"]
    for i, t in enumerate(traits):
        sh = lk[t]["share"]
        ok = sig(lk[t]["total"], q_lk[i]) and sig(lk[t]["via_readout"]) and math.isfinite(sh) and 0 <= sh <= 1
        tab.append(f"<tr><td class=l>{t}</td>{slope_cell(lk[t]['total'], q_lk[i])}"
                   f"{slope_cell(lk[t]['assoc'], q_lk[n_t + i])}{slope_cell(lk[t]['direct'], q_lk[2 * n_t + i])}"
                   f"<td>{f'{100 * sh:.0f}%' if ok else '<span class=muted>–</span>'}</td></tr>")
    tab.append("</table></div>" + games_note({r["cell"] for r in steered if r["trait"] in traits}))
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
    mcols = {"text": "text", "image": "image", "image − text": "diff"}
    i12, i_found, i_q = matrix("Readout slope (SD per unit c)", "trait", traits, list(mcols),
                               lambda t, m: obs_read[t][mcols[m]], games=[(g, m) for g in both_modes for m in ("text", "image")])
    j12, j_found, j_q = matrix("Steered agent's reward slope (SD per unit c)", "trait", traits, list(mcols),
                       lambda t, m: obs_rew[t][mcols[m]], games=[(g, m) for g in both_modes for m in ("text", "image")])
    D = "image − text"
    stronger_img = [t for t in traits if sig(i_found[(t, D)], i_q[(t, D)]) and i_found[(t, D)]["b"] > 0]
    weaker_img = [t for t in traits if sig(i_found[(t, D)], i_q[(t, D)]) and i_found[(t, D)]["b"] < 0]
    f12 = [f"Games played in both modes: {', '.join(both_modes) or 'none yet'}."]
    if stronger_img:
        f12.append(f"Steering moves the language <b>more</b> when the agent sees an image, for {len(stronger_img)} of "
                   f"{len(traits)} traits ({', '.join(stronger_img)}).")
    if weaker_img:
        f12.append(f"Less with images for: {', '.join(weaker_img)}.")
    if stronger_img or weaker_img:
        key.append(f12[1])
    mb = mode_baselines(rows)
    if mb:  # unsteered play itself: does the picture change how agents do, break, talk, and what they read?
        by_mode = {g: mean_ci([x for _, x in d["paired"]], [k for k, _ in d["paired"]]) for g, d in mb.items()}
        diff_g = [g for g, (m, h, _) in by_mode.items() if math.isfinite(h) and abs(m) > h]
        tin = [(d["text"]["tokens_in"], d["image"]["tokens_in"]) for d in mb.values()
               if math.isfinite(d["text"]["tokens_in"]) and math.isfinite(d["image"]["tokens_in"])]
        f12.append(f"<b>Without steering</b>, results differ between image and text play (95% CI of the paired "
                   f"difference excludes 0) in {len(diff_g)} of {len(mb)} games"
                   + (f" ({', '.join(diff_g)})" if diff_g else "") + "; unparseable answers are "
                   + f"{np.mean([d['image']['unparseable'] for d in mb.values()]):.1f}% with images vs "
                   + f"{np.mean([d['text']['unparseable'] for d in mb.values()]):.1f}% with text (mean over games)"
                   + (f"; agents read {np.mean([i for _, i in tin]):.0f} tokens per turn with images vs "
                      f"{np.mean([t for t, _ in tin]):.0f} with text" if tin else
                      "; token counts are not in this data yet") + ".")
    differ = [t for t in traits if sig(j_found[(t, D)], j_q[(t, D)])]
    if any(math.isfinite(obs_rew[t]["diff"].get("b", float("nan"))) for t in traits):
        f12.append("The steered agent's <b>result</b> responds differently to steering with images for "
                   + (", ".join(f"{t} ({'more' if obs_rew[t]['diff']['b'] > 0 else 'less'} reward per unit c)"
                                for t in differ) if differ else "no trait")
                   + " (reward table, image − text).")
    curves = mode_curves(rows, steered, scale, traits)
    mc = {"text": "var(--s1)", "image": "var(--s2)"}
    charts12 = ("<h3>Over the dose: text vs image</h3><div class=legend><span><i style='background:var(--s1)'></i>"
                "text</span><span><i style='background:var(--s2)'></i>image</span><span>games played both ways, persona "
                "vector (no side settings), bars: 95% CI</span></div><div class=charts>"
                + "".join(f"<div class=chart><h3>{html.escape(t)}: language</h3>"
                          f"{line_chart(curves[t]['readout'], t, f'{t} readout (SD)', colours=mc)}</div>"
                          f"<div class=chart><h3>{html.escape(t)}: result</h3>"
                          f"{line_chart(curves[t]['reward'], t, 'reward change (SD)', colours=mc)}</div>" for t in traits)
                + "</div>" + games_note([(g, m) for g in both_modes for m in ("text", "image")], "Games in these charts"))
    body.append(section(12, "Text vs image observations",
                        "Does seeing the game as a picture instead of text change how agents play at all, and does "
                        "steering act differently then?",
                        "Only games played in both modes. First the unsteered agents in each mode: their results (each "
                        "seat paired with itself in the same deal, since both modes use the same seeds), unparseable "
                        "answers, answer length, tokens read and generated per turn, game length. Then the steering "
                        "effect: the readout slope (section 2) and the steered agent's reward slope (section 9) for "
                        "text and image runs, diff = image − text; and both over the dose, text vs image.",
                        "A significant diff means the observation format changes how strongly the persona takes hold "
                        "(readout) or what it does to the agent's result (reward).",
                        f12, (modes_table(mb) if mb else "") + i12 + j12 + charts12))

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
              f"{line_chart({'vector': [(x, m, h) for x, (m, h, _) in ln_c.items()]}, 'length', 'characters')}</div></div>"
              + games_note({r["cell"] for r in co}, "Games in these charts"))
    f13 = []
    if 0.0 in ln_c and ln_c:
        hi_x, lo_x = max(ln_c), min(ln_c)
        f13.append(f"Answers are {ln_c[0.0][0]:.0f} characters long unsteered, {ln_c[hi_x][0]:.0f} at c = {hi_x:g} and "
                   f"{ln_c[lo_x][0]:.0f} at c = {lo_x:g}.")
    if fb_c:
        fb_steered = {x: v[0] for x, v in fb_c.items() if x != 0.0 and math.isfinite(v[0])}
        if 0.0 in fb_c and fb_steered:
            worst = max(fb_steered, key=fb_steered.get)
            f13.append(f"Unparseable answers: {fb_c[0.0][0]:.1f}% of turns unsteered, {min(fb_steered.values()):.1f}–"
                       f"{fb_steered[worst]:.1f}% steered (most at c = {worst:g})"
                       + (" — on average steering does not break answers more than the games' own parse failures."
                          if fb_steered[worst] <= fb_c[0.0][0] + 2 else " — strong steering breaks answers."))
        # pooled numbers hide that it depends on game and trait: the cells far from their game's unsteered rate
        per = defaultdict(list)
        for r in co:
            k_fb = (r["cell"], None) if r["condition"] == "none" else (r["cell"], r["trait"], r["coef"])
            per[k_fb].append(r["fallbacks"] / r["turns"])
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
    if fb_c and (broken or fixed):
        extra.append(
            "<b>Steering changes whether answers parse, in both directions.</b> "
            + (f"It breaks the format (≥ 25 points more unparseable turns than unsteered) in {len(broken)} game × "
               f"trait × c cells, worst {cell_name(broken[0][0][0])} × {broken[0][0][1]} at c = {broken[0][0][2]:g} "
               f"({100 * broken[0][1]:.0f}% vs {100 * base_fb[broken[0][0][0]]:.0f}%)" if broken else "")
            + ("; " if broken and fixed else "")
            + (f"it fixes it (≥ 20 points fewer) in {len(fixed)}, e.g. {cell_name(fixed[0][0][0])} × {fixed[0][0][1]} "
               f"at c = {fixed[0][0][2]:g} ({100 * fixed[0][1]:.0f}% vs {100 * base_fb[fixed[0][0][0]]:.0f}%)" if fixed
               else "")
            + ". Where a game's result depends on parsing, part of a reward effect is a format effect (section 13).")
    if 0.0 in ln_c and ln_c and math.isfinite(ln_c[0.0][0]):
        hi_x, lo_x = max(ln_c), min(ln_c)
        extra.append(f"<b>Steering changes how much agents write.</b> Answers average {ln_c[0.0][0]:.0f} characters "
                     f"unsteered, {ln_c[hi_x][0]:.0f} at c = {hi_x:g} and {ln_c[lo_x][0]:.0f} at c = {lo_x:g} (section 13; "
                     "per persona and setting in section 20 and in the steering mini report's recipe).")
    up = unparseable_cases(rows)
    more = [(k, v) for k, v in up["cases"].items() if _parse_flag(v[0], up["base"][k[0]], v[1]) == "more"]
    if more:
        by_set = defaultdict(int)
        for (cell, cond, c), _ in more:
            by_set[cond] += 1
        f13.append(f"Every setting, case by case (game × setting × strength, table below): {len(more)} of "
                   f"{len(up['cases'])} cases are {PARSE_GAP:g}+ points more unparseable than their game's base rate, by "
                   "setting: " + ", ".join(f"{c} {n}" for c, n in sorted(by_set.items(), key=lambda kv: -kv[1])) + ".")
    body.append(section(13, "Coherence cost: does steering break the agent?",
                        "Does strong steering make agents produce broken or unusually long/short answers?",
                        "Charts: steered agents (persona vector, all traits) by coefficient, unsteered agents at 0: the "
                        "share of turns that stayed unparseable after the retries, and the mean answer length. Tables: "
                        "every setting (random controls and prompts included), case by case, against each game's "
                        "unsteered base rate.",
                        "x = coefficient; a curve that shoots up at the ends = the model derails before the persona shows.",
                        f13, charts + unparseable_tables(rows)))

    late_take = []
    for part in (over_episode_section(rows, traits), actions_section(rows, traits, cells),
                 dose_response_section(steered, traits, cells, scale, own), footprint_section(rows, traits),
                 naming_section(rows, traits), prompt_vector_section(rows, traits, base, scale), length_section(rows, traits),
                 group_section(rows, traits, steered, scale), role_section(rows, traits, base, scale),
                 natural_section(rows, traits, steered, scale), risk_tempo_section(rows, traits, steered, scale),
                 modality_channel_section(rows, traits, base, scale, steered), push_section(rows, base, scale, norms)):
        late_take += part[0]
        body.append(part[1])
    pages = setting_pages(rows, traits, base, scale)  # each section's measure per setting, right under it
    body = [b + pages.get(b[b.index("id=s") + 4:b.index(">", b.index("id=s"))], "") if "id=s" in b[:40] else b
            for b in body]
    body.append(games_section(rows, traits, cells, man, eff, base, scale))

    # takeaways: what the sections add up to, written from their numbers
    take = takeaways(rows, traits, cells, bad, man, ratio, as_found, eff, g9_found, g9_q, p9_q, o_found, w_found, w_q,
                     ch, sides, beyond, corr, allm, small, entangled, readable, with_effect, through, hits, stronger_img,
                     weaker_img)
    at = len(take) - 1 if take and take[-1].startswith("<b>Read with care") else len(take)
    take[at:at] = late_take + extra  # sections 14-16, before the caveats
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
        f"<a href=#s{n}>{n}</a>" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "11b", 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)) + " · <a href=#games>per game</a></p>")
    coefs = sorted({r["coef"] for r in rows if r["coef"] is not None and r["via"] != "prompt"})
    ex = examples_section(logs_dir, traits, coefs) if logs_dir else ""
    return head + toc + take_html + setup_section(rows, sweep, traits, cells) + ex + "".join(body) + "</body></html>"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("agents_csv", help="output of scripts/summarize_runs.py")
    p.add_argument("--out", required=True)
    p.add_argument("--sweep", help="the sweep file (model, generation settings) for the setup section")
    p.add_argument("--model", help="the model key the sweep was run with (make_configs --model), if not its own")
    p.add_argument("--norms", help="scripts/vector_norms.py's output (default: vector_norms.json next to the CSV)")
    p.add_argument("--logs", nargs="+", help="the sweep's log dir(s) (<logging_dir>/<name>, and its fallback): adds "
                   "matched example answers")
    args = p.parse_args(argv)
    rows = load(args.agents_csv)
    if not rows:
        print("no rows")
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.make_configs import load_sweep
    sweep = load_sweep(args.sweep, args.model) if args.sweep else None
    norms_path = Path(args.norms or Path(args.agents_csv).with_name("vector_norms.json"))
    norms = json.loads(norms_path.read_text(encoding="utf-8")) if norms_path.exists() else None
    Path(args.out).write_text(build(rows, sweep, args.logs, norms), encoding="utf-8")
    print(f"report over {len(rows)} agent-episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Analysis report over every game of a sweep: one self-contained HTML page.

    python scripts/summarize_runs.py <logs>/main --out results/main/agents.csv
    python scripts/report.py results/main/agents.csv --out results/main/report.html

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
import html
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

STEERED_VIA = ("vector", "both")


# ---------------------------------------------------------------- data

def load(path) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            num = lambda k: float(r[k]) if r.get(k) not in (None, "") else None  # noqa: E731
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
    se = math.sqrt(var) if var > 0 else float("nan")
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


def detectability(rows, t):
    """{coefficient: (mean AUC, half CI, episodes)}: how well the steered agent's own-trait readout singles it
    out among the co-players of its episode (single-agent conditions). 0.5 = invisible to the monitor;
    for negative coefficients the steered agent is expected to read LOWER, and the AUC is oriented so."""
    episodes = defaultdict(list)
    for r in rows:
        if r["trait"] == t and r["select"] == "one" and r["via"] in STEERED_VIA and not r["random"] \
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
        se = math.sqrt(cov[j, j]) if cov[j, j] > 0 else float("nan")
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


# ---------------------------------------------------------------- HTML

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e6; --pos:#c0392b; --neg:#2c6fbb; --card:#f7f7f8; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececee; --muted:#9a9aa1; --line:#2c2c30;
  --card:#1e1e21; } }
body { background:var(--bg); color:var(--fg); font:14px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1200px;
  padding:16px; }
h1 { font-size:24px; } h2 { margin-top:40px; border-bottom:1px solid var(--line); padding-bottom:4px; }
p.note { color:var(--muted); max-width:900px; }
table { border-collapse:collapse; margin:8px 0 16px; font-variant-numeric:tabular-nums; }
th, td { border:1px solid var(--line); padding:3px 7px; text-align:right; } th { background:var(--card); }
td.l, th.l { text-align:left; }
.wrap { overflow-x:auto; }
.charts { display:grid; grid-template-columns:repeat(auto-fill, minmax(270px, 1fr)); gap:12px; }
.chart { background:var(--card); border-radius:8px; padding:8px; }
.chart h3 { margin:0 0 4px; font-size:14px; }
svg text { fill:var(--muted); font-size:10px; }
.sig { font-weight:700; }
"""


def fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def stars(q):
    return "" if not (q == q) else "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def heat(b, se):
    """Cell background by the slope's t value (red = up, blue = down)."""
    if not (se and math.isfinite(se) and se > 0 and math.isfinite(b)):
        return ""
    t = max(-1.0, min(1.0, (b / se) / 4))
    colour = "192,57,43" if t > 0 else "44,111,187"
    return f' style="background:rgba({colour},{abs(t) * 0.55:.2f})"'


def slope_cell(s, q=None):
    if not s or not math.isfinite(s.get("b", float("nan"))):
        return '<td></td>'
    tip = f'n={s["n"]}, episodes={s.get("clusters", "?")}, SE={fmt(s["se"])}, p={fmt(s["p"], 4)}' + \
          (f", q={fmt(q, 4)}" if q is not None else "")
    return f'<td{heat(s["b"], s["se"])} title="{tip}">{fmt(s["b"])}<span class="sig">{stars(q if q is not None else s["p"])}</span></td>'


def line_chart(series, title, width=260, height=170):
    """series: {name: [(x, mean, half_ci)]} -> small SVG chart."""
    pts = [p for s in series.values() for p in s if math.isfinite(p[1])]
    if not pts:
        return ""
    xs = [p[0] for p in pts]
    lo = min(p[1] - (p[2] if math.isfinite(p[2]) else 0) for p in pts)
    hi = max(p[1] + (p[2] if math.isfinite(p[2]) else 0) for p in pts)
    lo, hi = (lo - 1, hi + 1) if hi == lo else (lo, hi)
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    L, R, T, B = 34, 8, 8, 22
    sx = lambda x: L + (x - x0) / (x1 - x0) * (width - L - R)  # noqa: E731
    sy = lambda y: T + (hi - y) / (hi - lo) * (height - T - B)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<line x1="{L}" y1="{sy(0) if lo < 0 < hi else height - B}" x2="{width - R}" '
             f'y2="{sy(0) if lo < 0 < hi else height - B}" stroke="currentColor" stroke-opacity=".25"/>',
             f'<text x="2" y="{T + 8}">{hi:.2g}</text><text x="2" y="{height - B}">{lo:.2g}</text>',
             f'<text x="{L}" y="{height - 6}">{x0:g}</text><text x="{width - R - 14}" y="{height - 6}">{x1:g}</text>',
             f'<text x="{(L + width) / 2 - 20}" y="{height - 6}">coefficient</text>']
    colours = {"vector": "var(--pos)", "random": "var(--muted)", "prompt": "var(--neg)"}
    for name, s in series.items():
        s = sorted(p for p in s if math.isfinite(p[1]))
        c = colours.get(name, "currentColor")
        dash = ' stroke-dasharray="4 3"' if name == "random" else ""
        if len(s) > 1:
            parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6"{dash} points="'
                         + " ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m, _ in s) + '"/>')
        for x, m, h in s:
            if math.isfinite(h):
                parts.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(m - h):.1f}" y2="{sy(m + h):.1f}" '
                             f'stroke="{c}" stroke-opacity=".6"/>')
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(m):.1f}" r="2.6" fill="{c}"><title>c={x:g}: '
                         f'{m:.3f} ± {h:.3f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def cell_name(cell):
    group, obs = cell
    return group if obs == "text" else f"{group} ({obs})"


def build(rows) -> str:
    rows = zscored(rows)
    traits = sorted({r["trait"] for r in rows if r["trait"]})
    cells = sorted({r["cell"] for r in rows})
    base = baselines(rows)
    scale = reward_scale(rows)
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
           f"<meta name=viewport content='width=device-width, initial-scale=1'><title>Steering report</title>"
           f"<style>{CSS}</style></head><body><h1>Steering report</h1>",
           f"<p class=note>{len(rows)} agent-episodes, {len({r['cluster'] for r in rows})} episodes, "
           f"{len(cells)} games (game x observation), traits: {', '.join(traits)}. Hover a number for n, SE, p, q. "
           f"* q&lt;.05, ** q&lt;.01, *** q&lt;.001 (Benjamini-Hochberg within the table). Standard errors are "
           f"clustered by episode; p-values use a normal approximation, so treat cells with few episodes "
           f"with care.</p>"]

    # 1. coverage
    out.append("<h2>1. Coverage</h2><div class=wrap><table><tr><th class=l>game</th><th>runs</th><th>episodes</th>"
               "<th>unsteered episodes</th><th>truncated</th><th>unparseable answers</th></tr>")
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell]
        eps = {r["cluster"]: r["truncated"] for r in cr}
        turns, fb = sum(r["turns"] for r in cr), sum(r["fallbacks"] for r in cr)
        out.append(f"<tr><td class=l>{html.escape(cell_name(cell))}</td><td>{len({r['run_id'] for r in cr})}</td>"
                   f"<td>{len(eps)}</td><td>{len({r['cluster'] for r in cr if r['condition'] == 'none'})}</td>"
                   f"<td>{100 * sum(eps.values()) / max(len(eps), 1):.0f}%</td>"
                   f"<td>{100 * fb / max(turns, 1):.1f}%</td></tr>")
    out.append("</table></div>")

    # 2. manipulation check
    man = manipulation(rows, traits)
    out.append("<h2>2. Does steering move the trait? (manipulation check)</h2><p class=note>The <b>readout</b>: "
               "the steered agent's answers are fed to the UNsteered model; its hidden state after the monitor "
               "layer, averaged over the answer's tokens, is projected on the trait's unit vector. In SD units "
               "of the unsteered agents of the same game: 0 = a typical unsteered agent, +1 = one SD more of the "
               "trait. The <b>slope</b> is the change in readout per unit coefficient (e.g. 0.4: coefficient +2 "
               "reads about +0.8 SD). Red: the persona vector; grey dashed: a random direction of the same norm "
               "(should stay flat); blue points at ±1: the persona prompt (positive / negative instruction). "
               "Game fixed effects.</p>")
    q_man = bh([man[t]["vector"]["p"] for t in traits] + [man[t]["random"]["p"] for t in traits])
    out.append("<div class=wrap><table><tr><th class=l>trait</th><th>vector slope</th><th>random slope</th>"
               "<th>prompt + (vs unsteered)</th><th>prompt − (vs unsteered)</th></tr>")
    for i, t in enumerate(traits):
        pp, pn = man[t]["prompt"][1], man[t]["prompt"][-1]
        out.append(f"<tr><td class=l>{t}</td>{slope_cell(man[t]['vector'], q_man[i])}"
                   f"{slope_cell(man[t]['random'], q_man[len(traits) + i])}"
                   f"<td>{fmt(pp[0])} ± {fmt(pp[1])}</td><td>{fmt(pn[0])} ± {fmt(pn[1])}</td></tr>")
    out.append("</table></div><div class=charts>")
    for t in traits:
        curve = man[t]["curve"]
        series = {"vector": [(x, m, h) for (kind, x), (m, h, _) in curve.items() if kind == "vector"],
                  "random": [(x, m, h) for (kind, x), (m, h, _) in curve.items() if kind == "random"]
                  + ([(0.0, *curve[("vector", 0.0)][:2])] if ("vector", 0.0) in curve else []),
                  "prompt": [(s, *man[t]["prompt"][s][:2]) for s in (1, -1)]}
        out.append(f"<div class=chart><h3>{t}</h3>{line_chart(series, t)}</div>")
    out.append("</div>")
    out.append(_game_table("Vector slope of the trait readout, per game", traits, cells,
                           {t: man[t]["per_game"] for t in traits}))

    # 3. specificity: does steering one trait move the others?
    out.append("<h2>3. Specificity: what else moves?</h2><p class=note>Rows: the trait steered (persona vector); "
               "columns: the trait read out from the steered agent's answers. Slope per unit coefficient, SD units. "
               "A clean vector moves its own trait (diagonal) and little else; off-diagonal cells show which "
               "traits come along (e.g. steering 'evil' also lowering 'empathy').</p>")
    readouts = sorted({u for r in rows for u in r["mon"]})
    out.append(_matrix("", "steered / read", traits, readouts,
                       lambda t, u: readout_slope(rows, t, lambda r: is_vector(r), readout=u)))

    # 4. detectability: can the monitor single out the steered agent?
    out.append("<h2>4. Detectability: can a monitor tell who is steered?</h2><p class=note>Episodes where one "
               "agent is steered: the share of its co-players whose readout of the steered trait is lower than "
               "the steered agent's (higher, for negative coefficients). 0.5 = the monitor cannot tell; 1 = the "
               "steered agent always stands out. Mean over episodes ± 95% CI.</p>")
    det = {t: detectability(rows, t) for t in traits}
    coefs = sorted({c for d in det.values() for c in d})
    out.append("<div class=wrap><table><tr><th class=l>trait</th>" + "".join(f"<th>c={c:g}</th>" for c in coefs)
               + "</tr>")
    for t in traits:
        tds = []
        for c in coefs:
            m, h, n = det[t].get(c, (float("nan"), float("nan"), 0))
            shade = (f' style="background:rgba(192,57,43,{max(0.0, min(1.0, (m - 0.5) * 2)) * 0.55:.2f})"'
                     if math.isfinite(m) else "")
            ci = "" if not math.isfinite(h) else f" ±{h:.2f}"
            tds.append(f'<td{shade} title="episodes={n}">{fmt(m, 2)}{ci}</td>')
        out.append(f"<tr><td class=l>{t}</td>{''.join(tds)}</tr>")
    out.append("</table></div>")

    # 5. asymmetry: amplify vs suppress
    out.append("<h2>5. Amplify vs suppress</h2><p class=note>Readout slope fitted on the positive coefficients "
               "only and on the negative ones only (each with the unsteered agents at 0). Suppression is often "
               "weaker: an agent with little of the trait has little to remove.</p>")
    out.append(_matrix("", "trait", traits, ["positive coefficients", "negative coefficients"],
                       lambda t, side: readout_slope(rows, t, lambda r: is_vector(r) and
                                                     ((r["coef"] > 0) if side.startswith("pos") else (r["coef"] < 0)))))

    # 6. channels
    out.append("<h2>6. Persona prompt vs vector vs both</h2><p class=note>Readout of the steered agent in SD units. "
               "Prompt: the trait's instruction in the system prompt (positive / negative). Vector at max ±: the "
               "mean readout of vector-steered agents at the largest coefficients. Slopes: the vector alone and "
               "the prompt plus the vector (condition both_one), per unit coefficient.</p>")
    out.append("<div class=wrap><table><tr><th class=l>trait</th><th>prompt +</th><th>prompt −</th>"
               "<th>vector at max +</th><th>vector at max −</th><th>vector-only slope</th>"
               "<th>prompt+vector slope</th></tr>")

    def mean_cell(v):
        return f"<td>{fmt(v[0], 2)} ± {fmt(v[1], 2)}</td>" if v else "<td></td>"

    for t in traits:
        curve, pr = man[t]["curve"], man[t]["prompt"]
        vx = [x for (kind, x) in curve if kind == "vector" and x != 0]
        hi = curve.get(("vector", max(vx))) if vx else None
        lo = curve.get(("vector", min(vx))) if vx else None
        both = readout_slope(rows, t, lambda r: r["steered"] and r["via"] == "both" and not r["random"])
        only = readout_slope(rows, t, lambda r: r["steered"] and r["via"] == "vector" and not r["random"])
        out.append(f"<tr><td class=l>{t}</td>{mean_cell(pr[1])}{mean_cell(pr[-1])}{mean_cell(hi)}{mean_cell(lo)}"
                   f"{slope_cell(only)}{slope_cell(both)}</tr>")
    out.append("</table></div>")

    # 7. who is steered
    conds = sorted({r["condition"] for r in rows if is_vector(r)})
    out.append("<h2>7. Who is steered: one, a fraction, all, a team</h2><p class=note>Per steering condition "
               "(rows) and trait (columns). <b>Readout</b>: the steered agents' own-trait readout slope (SD units). "
               "<b>Own reward</b>: the steered agents' paired reward slope. <b>Group welfare</b>: the sum of ALL "
               "agents' rewards in the episode minus that of the unsteered episode with the same seed, in SD "
               "units, per unit coefficient: does steering one, half or all agents make the group better off? "
               "evil_one / good_one / evil_team: hidden-role games, by side.</p>")
    out.append(_matrix("Readout of the steered agents", "condition", conds, traits,
                       lambda c, t: readout_slope(rows, t, lambda r: is_vector(r) and r["condition"] == c)))
    own = paired(rows, base, lambda r: is_vector(r))
    out.append(_matrix("Own reward (SD units)", "condition", conds, traits,
                       lambda c, t: _diff_slope([r for r in own if r["condition"] == c and r["trait"] == t], scale)))
    wel = welfare([r for r in rows if r["condition"] != "none"], [r for r in rows if r["condition"] == "none"])
    out.append(_matrix("Group welfare (SD units)", "condition", conds, traits,
                       lambda c, t: _diff_slope([r for r in wel if r["condition"] == c and r["trait"] == t], None)))

    # 8. contagion
    out.append("<h2>8. Contagion: do the others start to sound like the steered agent?</h2><p class=note>The "
               "UNsteered co-players' readout of the steered trait, per unit coefficient (SD units), in episodes "
               "where one agent or a fraction is steered. Positive: the trait spreads through the conversation "
               "(or the others answer in kind).</p>")
    partial = sorted({r["condition"] for r in rows if is_vector(r) and r["select"] in ("one", "count", "fraction")})
    out.append(_matrix("", "condition", partial, traits,
                       lambda c, t: readout_slope(rows, t, lambda r: r["condition"] == c and not r["steered"])))

    # 9-11. outcomes per game, spillover on reward, evil team
    steered = paired(rows, base, lambda r: is_vector(r) and r["team_cond"] is None)
    eff = effect_table(steered, traits, scale)
    out.append("<h2>9. Outcomes per game: the steered agent's reward</h2><p class=note>Paired effect: the steered "
               "agent's reward minus the reward of the same seat in the unsteered episode with the same seed "
               "(same deal and roles), per unit coefficient. Pooled: rewards in units of each game's unsteered "
               "reward SD, game fixed effects. Per game: raw reward units.</p>")
    out.append(_pooled_table(eff, traits))
    out.append(_game_table("Reward slope of the steered agent, per game", traits, cells, eff))
    spill = paired(rows, base, lambda r: not r["steered"] and r["trait"] and r["select"] in ("one", "count", "fraction")
                   and r["team_cond"] is None and r["via"] in STEERED_VIA and not r["random"])
    eff_s = effect_table(spill, traits, scale)
    out.append("<h2>10. Spillover on the others' reward</h2><p class=note>The same paired effect for the UNsteered "
               "agents of episodes where one agent (or a fraction) is steered.</p>")
    out.append(_pooled_table(eff_s, traits))
    out.append(_game_table("Reward slope of the unsteered co-players, per game", traits, cells, eff_s))
    team_rows = paired(rows, base, lambda r: r["team_cond"] == "evil" and r["team"] == "evil" and is_vector(r))
    if team_rows:
        eff_t = effect_table(team_rows, traits, scale)
        out.append("<h2>11. Steering the evil team (hidden-role games)</h2><p class=note>Paired reward effect "
                   "of the evil side when all of it is steered.</p>")
        out.append(_pooled_table(eff_t, traits))
        out.append(_game_table("Evil-team reward slope, per game", traits,
                               sorted({r["cell"] for r in team_rows}), eff_t))

    # 11b. does the readout go with the outcome?
    out.append("<h2>11b. Does the trait in the language go with the outcome?</h2><p class=note>Steered agents "
               "(persona vector). <b>Coefficient → outcome</b>: the total effect of steering on the agent's paired "
               "reward (SD units per unit coefficient). <b>Readout → outcome</b>: across these agents, how much "
               "the outcome changes per +1 SD of trait readout (do agents that came out more sarcastic do "
               "better or worse?). <b>Coefficient → outcome, readout fixed</b>: the steering effect that the "
               "readout does not account for. <b>Share via readout</b>: 1 − fixed / total, shown only when the "
               "total effect is significant. This is a correlational split, not proof that the language causes "
               "the outcome: the vector can change behaviour (votes, bids, moves) through other routes.</p>")
    lk = {t: link(steered, t, scale) for t in traits}
    q_lk = bh([lk[t]["total"]["p"] for t in traits] + [lk[t]["assoc"]["p"] for t in traits]
              + [lk[t]["direct"]["p"] for t in traits])
    out.append("<div class=wrap><table><tr><th class=l>trait</th><th>coefficient → outcome</th>"
               "<th>readout → outcome</th><th>coefficient → outcome, readout fixed</th><th>share via readout</th></tr>")
    n_t = len(traits)
    for i, t in enumerate(traits):
        sh = lk[t]["share"]
        out.append(f"<tr><td class=l>{t}</td>{slope_cell(lk[t]['total'], q_lk[i])}"
                   f"{slope_cell(lk[t]['assoc'], q_lk[n_t + i])}{slope_cell(lk[t]['direct'], q_lk[2 * n_t + i])}"
                   f"<td>{'' if not math.isfinite(sh) else f'{100 * sh:.0f}%'}</td></tr>")
    out.append("</table></div>")
    out.append(_matrix("Readout → outcome per game (raw reward per +1 SD of readout)", "game", cells, traits,
                       lambda c, t: slope([r["mon"][t] for r in steered if r["cell"] == c and r["trait"] == t
                                           and t in r["mon"]],
                                          [r["diff"] for r in steered if r["cell"] == c and r["trait"] == t
                                           and t in r["mon"]],
                                          [r["cluster"] for r in steered if r["cell"] == c and r["trait"] == t
                                           and t in r["mon"]])))

    # 12. text vs image
    out.append("<h2>12. Text vs image observations</h2><p class=note>The same games played with the state as "
               "text and as a picture (only games run in both modes). Does steering act differently when the "
               "agent sees an image? Difference = image slope minus text slope.</p>")
    obs_read, obs_rew, both_modes = {}, {}, []
    for t in traits:
        obs_read[t], both_modes = obs_compare(rows, t, lambda sub, t: readout_slope(sub, t, lambda r: is_vector(r)))
        obs_rew[t], _ = obs_compare(steered, t,
                                    lambda sub, t: _diff_slope([r for r in sub if r["trait"] == t], scale))
    out.append(f"<p class=note>games in both modes: {', '.join(both_modes) or 'none yet'}</p>")
    out.append(_matrix("Readout slope", "trait", traits, ["text", "image", "diff"], lambda t, m: obs_read[t][m]))
    out.append(_matrix("Steered agent's reward slope (SD units)", "trait", traits, ["text", "image", "diff"],
                       lambda t, m: obs_rew[t][m]))

    # 13. coherence cost
    out.append("<h2>13. Coherence cost: does steering break the agent?</h2><p class=note>Steered agents (persona "
               "vector, all traits) by coefficient, with the unsteered agents at 0: the share of turns whose answer "
               "could not be parsed after all retries, and the mean answer length. Strong steering often derails "
               "the model before it changes behaviour.</p>")
    co = [r for r in rows if r["turns"] and (is_vector(r) or r["condition"] == "none")]
    fb, ln = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    for r in co:
        x = r["coef"] if r["condition"] != "none" else 0.0
        fb[x][0].append(100 * r["fallbacks"] / r["turns"])
        fb[x][1].append(r["cluster"])
        if r["chars"] is not None:
            ln[x][0].append(r["chars"])
            ln[x][1].append(r["cluster"])
    out.append("<div class=charts>")
    for name, d in (("unparseable answers (%)", fb), ("answer length (characters)", ln)):
        series = {"vector": [(x, *mean_ci(*v)[:2]) for x, v in d.items()]}
        out.append(f"<div class=chart><h3>{name}</h3>{line_chart(series, name)}</div>")
    out.append("</div>")

    def fallback_slope(t, _):
        sel = [r for r in co if r["condition"] == "none" or r["trait"] == t]
        return slope([abs(r["coef"]) if r["condition"] != "none" else 0.0 for r in sel],
                     [100 * r["fallbacks"] / r["turns"] for r in sel], [r["cluster"] for r in sel])

    out.append(_matrix("Unparseable-answer rate (% points) per unit |coefficient|", "trait", traits, ["slope"],
                       fallback_slope))
    out.append("</body></html>")
    return "\n".join(out)


def _diff_slope(rs, scale):
    """Slope of a paired difference over the coefficient (scaled by the game's unsteered SD if `scale`)."""
    rs = [r for r in rs if r["coef"] is not None]
    return slope([r["coef"] for r in rs], [r["diff"] / (scale.get(r["cell"], 1.0) if scale else 1.0) for r in rs],
                 [r["cluster"] for r in rs], [r["cell"] for r in rs])


def _matrix(title, corner, row_keys, col_keys, get):
    """A table of slopes: rows x columns, BH-corrected across all its cells."""
    keys = [(a, b) for a in row_keys for b in col_keys]
    found = {k: get(*k) for k in keys}
    q = dict(zip(keys, bh([found[k]["p"] if found[k] else float("nan") for k in keys])))
    out = [f"<h3>{html.escape(title)}</h3>" if title else "",
           f"<div class=wrap><table><tr><th class=l>{html.escape(corner)}</th>"
           + "".join(f"<th>{html.escape(str(c))}</th>" for c in col_keys) + "</tr>"]
    for a in row_keys:
        out.append(f"<tr><td class=l>{html.escape(cell_name(a) if isinstance(a, tuple) else str(a))}</td>"
                   + "".join(slope_cell(found[(a, b)], q[(a, b)]) for b in col_keys) + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def _pooled_table(eff, traits):
    q = bh([eff[t]["pooled"]["p"] for t in traits])
    head = "".join(f"<th>{t}</th>" for t in traits)
    cells = "".join(slope_cell(eff[t]["pooled"], q[i]) for i, t in enumerate(traits))
    return f"<div class=wrap><table><tr><th class=l>pooled (SD units)</th>{head}</tr><tr><td class=l>all games</td>{cells}</tr></table></div>"


def _game_table(title, traits, cells, per_trait):
    keys = [(c, t) for c in cells for t in traits]
    found = [per_trait[t].get(c) for c, t in keys]
    q = bh([s["p"] if s else float("nan") for s in found])
    qs = dict(zip(keys, q))
    rows = [f"<h3>{html.escape(title)}</h3><div class=wrap><table><tr><th class=l>game</th>"
            + "".join(f"<th>{t}</th>" for t in traits) + "</tr>"]
    for c in cells:
        rows.append(f"<tr><td class=l>{html.escape(cell_name(c))}</td>"
                    + "".join(slope_cell(per_trait[t].get(c), qs[(c, t)]) for t in traits) + "</tr>")
    rows.append("</table></div>")
    return "".join(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("agents_csv", help="output of scripts/summarize_runs.py")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    rows = load(args.agents_csv)
    if not rows:
        print("no rows")
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(build(rows), encoding="utf-8")
    print(f"report over {len(rows)} agent-episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

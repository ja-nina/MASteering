"""Analysis report over every game of a sweep: one self-contained HTML page.

    python scripts/summarize_runs.py <logs>/main --out results/main/agents.csv
    python scripts/report.py results/main/agents.csv --out results/main/report.html

Reads the agent-episode rows of summarize_runs.py and answers, per trait and per game:
  1. coverage        episodes per game, truncated episodes, unparseable answers
  2. manipulation    does steering move the steered agent's trait readout (monitor projection)?
                     dose-response slope over the coefficient (unsteered agents of the game at 0),
                     the norm-matched random direction (should be flat), the prompt persona (+/-)
  3. behaviour       paired effect on reward: the steered agent's reward minus the reward of the
                     same seat in the unsteered episode with the same seed (same deal and roles),
                     regressed on the coefficient
  4. spillover       the same paired effect for the UNsteered agents of single-agent conditions
  5. teams           steering the whole evil team (hidden-role games): the evil side's reward
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
                "truncated": r.get("truncated") == "True",
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
    out.append("<h2>2. Does steering move the trait? (manipulation check)</h2><p class=note>The steered agent's "
               "monitor readout of its own trait (its answers read by the unsteered model, projected on the unit "
               "trait vector), minus the unsteered agents' mean in the same game. Red: the persona vector; grey "
               "dashed: a random direction of the same norm (should stay flat); blue points at ±1: the persona "
               "prompt (positive / negative instruction). Slopes per unit coefficient, game fixed effects.</p>")
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

    # 3. behaviour: steered agents' paired reward effect
    steered = paired(rows, base, lambda r: r["steered"] and r["via"] in STEERED_VIA and not r["random"]
                     and r["team_cond"] is None)
    eff = effect_table(steered, traits, scale)
    out.append("<h2>3. Does the steered agent's outcome change?</h2><p class=note>Paired effect: the steered "
               "agent's reward minus the reward of the same seat in the unsteered episode with the same seed "
               "(same deal and roles), regressed on the coefficient. Pooled: rewards in units of each game's "
               "unsteered reward SD, game fixed effects. Per game: raw reward units per unit coefficient.</p>")
    out.append(_pooled_table(eff, traits))
    out.append(_game_table("Reward slope of the steered agent, per game", traits, cells, eff))

    # 4. spillover
    spill = paired(rows, base, lambda r: not r["steered"] and r["trait"] and r["select"] in ("one", "count", "fraction")
                   and r["team_cond"] is None and r["via"] in STEERED_VIA and not r["random"])
    eff_s = effect_table(spill, traits, scale)
    out.append("<h2>4. Spillover to the other agents</h2><p class=note>The same paired effect for the UNsteered "
               "agents of episodes where one agent (or a fraction) is steered.</p>")
    out.append(_pooled_table(eff_s, traits))
    out.append(_game_table("Reward slope of the unsteered co-players, per game", traits, cells, eff_s))

    # 5. teams
    team_rows = paired(rows, base, lambda r: r["team_cond"] == "evil" and r["team"] == "evil" and r["steered"]
                       and r["via"] in STEERED_VIA and not r["random"])
    if team_rows:
        eff_t = effect_table(team_rows, traits, scale)
        out.append("<h2>5. Steering the evil team (hidden-role games)</h2><p class=note>Paired reward effect "
                   "of the evil side when all of it is steered.</p>")
        out.append(_pooled_table(eff_t, traits))
        out.append(_game_table("Evil-team reward slope, per game", traits,
                               sorted({r["cell"] for r in team_rows}), eff_t))
    out.append("</body></html>")
    return "\n".join(out)


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

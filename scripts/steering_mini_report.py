"""The steering mini report: what steering does, in plain terms, per game and steering vector.

    python scripts/steering_mini_report.py results/main/agents.csv.gz \\
        --logs <logging_dir>/main [fallback/main] --out results/main/steering_mini_report.html

The same agent rows as scripts/report.py, but no slopes, standard deviations or q-values: everything is drawn as
line plots over the steering strength c, one line per setting (how many and which players are steered), or said
in words ("much more sarcastic", "no clear change") with a confidence marker; results in the game's own terms
(wins, points). Contents: how to read it and how the monitor works; headline findings; a recipe table; prompt,
vector and both, and whether they are synergistic; text vs image; language vs action; hidden roles by who
carries the persona; a gallery of answers steering breaks or fixes; matched examples; per game and steering
vector, every setting.
"""
from __future__ import annotations

import argparse
import html
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import report as R  # noqa: E402

SETTINGS = ("one", "half", "all", "both_one", "good_one", "evil_one", "evil_team", "good_team", "all_random",
            "prompt_one", "prompt_all", "evil_team_prompt", "all_random_L16", "all_random_L32")
SETTING_WORDS = {
    "one": "one agent, vector", "half": "half the agents, vector", "all": "every agent, vector",
    "both_one": "one agent, vector + prompt", "good_one": "one honest player, vector",
    "evil_one": "one deceiver, vector", "evil_team": "every deceiver, vector", "good_team": "every honest player, vector",
    "all_random": "every agent, random direction (control)", "prompt_one": "one agent, prompt",
    "prompt_all": "every agent, prompt", "evil_team_prompt": "every deceiver, prompt", "unsteered": "unsteered",
    "all_random_L16": "every agent, random direction at layer 16 (control)",
    "all_random_L32": "every agent, random direction at layer 32 (control)"}
# a prompt setting shares its vector counterpart's colour (drawn as squares); the random control is dashed grey
COLOUR = {**{c: f"var(--s{i + 1})" for i, c in enumerate(R.SIDE_SETTINGS)}, "all_random": "var(--rnd)",
          "prompt_one": "var(--s1)", "prompt_all": "var(--s3)", "evil_team_prompt": "var(--s7)",
          "all_random_L16": "var(--s2)", "all_random_L32": "var(--s6)",  # dashed too (random in the name)
          "unsteered": "currentColor"}
MARK = {3: "●●●", 2: "●●○", 1: "●○○", 0: "○○○"}
MARK_TIP = {3: "solid", 2: "tentative", 1: "no clear change", 0: "too little data"}


# ---------------------------------------------------------------- judging an effect, in words

def judge(values, clusters):
    """(mean, level) of a paired or standardised effect: level 3 = solid (p < .001, >= 10 independent groups),
    2 = tentative (p < .01, >= 5), 1 = no clear change, 0 = too little data. Many effects are judged, so a
    'tentative' one can be chance; 'solid' rarely is."""
    m, se, n = R._episode_mean(values, clusters)
    if n < 5 or not math.isfinite(m):
        return m, 0
    z = abs(m / se) if math.isfinite(se) and se > 0 else (float("inf") if m else 0.0)
    return m, 3 if z >= 3.29 and n >= 10 else 2 if z >= 2.58 else 1


def number(v, unit=""):
    """A plain number: whole percents, else 3 significant digits; 0 for rounding noise."""
    if unit == "%":
        return f"{v:.0f}%" if abs(v) >= 0.5 else "0%"  # not "-0%"
    return "0" if abs(v) < 0.005 else f"{v:.3g}"


def mark(level):
    return f"<span class=mk title='{MARK_TIP[level]}'>{MARK[level]}</span>"


def words(m, level, trait):
    """A language effect (readout in SD of unsteered agents, not shown) in words."""
    if level == 0:
        return f"<span class=muted>too little data</span> {mark(0)}"
    if level == 1:
        return f"<span class=muted>no clear change</span> {mark(1)}"
    size = "much " if abs(m) >= 2 else "" if abs(m) >= 0.5 else "a bit "
    return f"{size}{'more' if m > 0 else 'less'} {html.escape(trait)} {mark(level)}"


def language(rs, t):
    """(words, mean, level) for the readout of trait t of the steered agents rs ('not measured' for a sweep played
    without the monitor)."""
    if not MONITORED:
        return "<span class=muted>not measured</span>", float("nan"), 0
    rs = [r for r in rs if t in r["mon"] and math.isfinite(r["mon"][t])]
    m, level = judge([r["mon"][t] for r in rs], [r["cluster"] for r in rs])
    return words(m, level, t), m, level


def result_level(rs):
    """(mean paired change, level) of the steered agents' result; clusters = seeds (one unsteered game each)."""
    rs = [r for r in rs if math.isfinite(r["diff"])]
    return judge([r["diff"] for r in rs], [(r["cell"], r["episode"]) for r in rs])


def win_games(rows):
    """Game cells whose reward is a win or a loss (0/1 or -1/1): their results read as wins."""
    vals = defaultdict(set)
    for r in rows:
        if r["reward"] is not None:
            vals[r["cell"]].add(round(r["reward"], 6))
    return {c for c, v in vals.items() if v <= {0.0, 1.0} or v <= {-1.0, 1.0}}


def curve(rs, value, cluster=lambda r: r["cluster"]):
    """[(c, mean, half 95% CI)] of value(r) per steering coefficient, plus the unsteered point (0, 0)."""
    pts = [(c, *R.mean_ci([value(r) for r in v], [cluster(r) for r in v])[:2])
           for c, v in sorted(R._by([r for r in rs if r["coef"]], lambda r: r["coef"]).items())]
    return pts + [(0.0, 0.0, float("nan"))]


def charts_row(panels, ylab):
    """Line charts on one shared y axis: panels = [(title, {setting: points})]."""
    ext = [v for _, series in panels for pts in series.values() for _, m, h in pts if math.isfinite(m)
           for v in (m - (h if math.isfinite(h) else 0), m + (h if math.isfinite(h) else 0))]
    span = (min(ext + [0]), max(ext + [0])) if ext else None
    return "".join(f"<div class=chart><h3>{html.escape(title)}</h3>"
                   + R.line_chart(series, title, ylab, colours=COLOUR, span=span,
                                  notes={c: SETTING_WORDS.get(c, c) for c in series}) + "</div>"
                   for title, series in panels if series)


def legend(sets):
    return ("<div class=legend>" + "".join(
        f"<span><i style='background:{COLOUR.get(c, 'currentColor')}" + (";height:7px;width:7px" if "prompt" in c else "")
        + f"'></i>{html.escape(SETTING_WORDS.get(c, c))}</span>" for c in sets)
        + "<span>squares: the prompt (the opposite instruction at c = −1, the instruction at +1); dashed: random "
          "direction; bars: 95% CI; c = 0: unsteered</span></div>")


# ---------------------------------------------------------------- the parts

def headline(by, paired_by):
    """Plain findings at full strength (c = ±2), with markers."""
    lang = n_lang = outc = n_out = 0
    for (cell, t, c), rs in by.items():
        if c != "one":
            continue
        for s in (1, -1):
            st = [r for r in rs if r["steered"] and r["coef"] == 2 * s]
            if st:
                _, m, level = language(st, t)
                n_lang += 1
                lang += level >= 2 and m * s > 0
            pr = [r for r in paired_by.get((cell, t, c), []) if r["steered"] and r["coef"] == 2 * s]
            if pr:
                n_out += 1
                outc += result_level(pr)[1] >= 2
    games = sorted({cell for (cell, t, c) in by if c == "one"})
    if not MONITORED:
        return [f"One agent steered at full strength (c = ±2): its result in the game changes clearly in {outc} of "
                f"{n_out} game × persona × direction cases. {mark(3)} Its language was not measured (played without the "
                "monitor)." + R.games_note(games, "Games counted")]
    return [f"<b>Steering changes what agents say almost everywhere, what they win far less often.</b> One agent "
            f"steered at full strength (c = ±2): its language moves the intended way in {lang} of {n_lang} game × "
            f"persona × direction cases; its result in the game changes clearly in {outc} of {n_out}. {mark(3)}"
            + R.games_note(games, "Games counted")]


def recipe(rows, traits, by_trait, paired_by, cells):
    """Per persona and direction, one agent steered with the vector only: how far its language moves at full
    strength, the mildest strength that works, what full strength costs, and in which games it changes the agent's
    result. Prompt and both are compared in synergy()."""
    none = [r for r in rows if r["condition"] == "none"]
    fb0 = 100 * sum(r["fallbacks"] for r in none) / max(1, sum(r["turns"] for r in none))
    len0 = {c: np.mean([r["chars"] for r in rs if r["chars"]]) for c, rs in R._by(none, lambda r: r["cell"]).items()}
    out = ["<h2 id=recipe>Recipe: steering one agent with the vector, and what it costs</h2><p class=muted>One "
           "agent steered with the vector only (no prompt); every column is about that agent, at full strength "
           "(c = ±2) except the weakest strength. Language: how far the agent's language moves; unparseable turns: "
           f"turns the game could not parse (unsteered: {fb0:.1f}%); results: games where the agent's own result "
           "changes clearly, against the same seat in unsteered games with the same seeds. Prompt and both together: "
           "<a href=#synergy>prompt, vector, both</a>.</p><div class=wrap><table><tr><th class=l>"
           "persona</th><th>language (vector, c = ±2)</th><th>weakest vector strength with a "
           "clear effect on the language</th><th>unparseable turns (vector, c = ±2)</th><th>answer length vs "
           "unsteered (vector, c = ±2)</th><th class=l>games where the agent's result gets better (vector, c = ±2)</th>"
           "<th class=l>games where it gets worse (vector, c = ±2)</th></tr>"]
    for t in traits:
        for s in (1, -1):
            rs = [r for r in by_trait.get(("one", t), []) if r["steered"] and r["coef"] == 2 * s]
            if not rs:
                continue
            mild = next((c for c in (0.5, 1, 1.5, 2) if (lambda v: v[2] >= 2 and v[1] * s > 0)(
                language([r for r in by_trait.get(("one", t), []) if r["steered"] and r["coef"] == c * s], t))), None)
            fb = 100 * sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs))
            ratio = np.mean([r["chars"] / len0[r["cell"]] for r in rs if r["chars"] and len0.get(r["cell"])] or [1])
            shape = ("about as long" if 0.8 <= ratio <= 1.25 else f"{ratio:.1f}× longer" if ratio > 1
                     else f"{1 / ratio:.1f}× shorter")
            better, worse = [], []
            for cell in cells:
                pr = [r for r in paired_by.get((cell, t, "one"), []) if r["steered"] and r["coef"] == 2 * s]
                if pr:
                    m, level = result_level(pr)
                    if level >= 2:
                        (better if m > 0 else worse).append(f"{html.escape(R.cell_name(cell))} {mark(level)}")
            none_ = "<span class=muted>none clearly</span>"
            out.append(f"<tr><td class=l>{'more' if s > 0 else 'less'} {t}</td><td>{language(rs, t)[0]}</td>"
                       f"<td>{f'c = {mild * s:+g}' if mild else 'none clearly'}</td>"
                       f"<td>{fb:.1f}%{' ⚠' if fb > fb0 + 5 else ''}</td><td>{shape}</td>"
                       f"<td class=l>{', '.join(better) or none_}</td><td class=l>{', '.join(worse) or none_}</td></tr>")
    games = sorted({r["cell"] for t in traits for r in by_trait.get(("one", t), []) if r["steered"]})
    return "".join(out) + "</table></div>" + R.games_note(games)


def _together(parts):
    """Verdict on both − (vector + prompt) from the three (mean, SE, n): more than the sum, adds up, less."""
    se = math.sqrt(sum(x[1] ** 2 for x in parts)) if all(math.isfinite(x[1]) for x in parts) else 0
    z = (parts[0][0] - parts[1][0] - parts[2][0]) / se if se > 0 else float("nan")
    return ("<span class=muted>too little data</span>" if not math.isfinite(z) else
            "more than the sum" if z > 2.58 else "less than the sum" if z < -2.58 else "adds up")


def synergy(traits, by_trait, paired_by, modes_games, per_game):
    """Prompt, vector, both at c = ±1 (the prompt has no strength): the language, and whether the combination is
    more than the sum of the two (synergy), the sum (they add), or less (they overlap), for language and result;
    over all games, per observation mode on the games played both ways, and game by game (`per_game`: report.py's
    synergy_cells). The pooled language verdict pools the games' own synergies (raw readout, comparable across games)
    with a random-effects model: the games differ in how prompt and vector combine, and a pool that ignored that would
    call nearly every case more or less than the sum."""
    out = ["<h2 id=synergy>Prompt, vector, or both: do they reinforce each other?</h2><p class=muted>One agent, "
           "the vector at c = ±1 (the prompt has no strength). Each is compared with the same unsteered games, so if "
           "the two simply add, 'both' = vector + prompt. Verdict: <b>more than the sum</b> (synergy), <b>adds up</b>, "
           "or <b>less than the sum</b> (they overlap or compete; on the language this can also be a ceiling). "
           "'All games' pools every game; 'text' and 'image' use only the games played both ways, so the two "
           "compare the same games. 'Games' counts the verdict game by game (more / adds up / less): pooling "
           "averages over games, so a reinforcement in one game and an overlap in another can cancel out. The pooled "
           "language verdicts allow for the games differing (random effects over the per-game results), so they say "
           "'more' or 'less' only where that holds across games.</p>"
           "<div class=wrap><table><tr><th class=l rowspan=2>persona</th><th rowspan=2>vector</th><th rowspan=2>prompt"
           "</th><th rowspan=2>both</th><th colspan=4>language: together</th><th colspan=4>result: together</th></tr>"
           "<tr>" + ("<th>pooled over all games</th><th>text, games played both ways</th><th>image, games played "
                     "both ways</th><th>number of games: more than the sum / adds up / less than the sum</th>") * 2
           + "</tr>"]
    pick = lambda cond, t, s, c=None, m=None: [
        r for r in by_trait.get((cond, t), []) if r["steered"] and r["coef"] * s > 0
        and (c is None or abs(r["coef"]) == c) and (m is None or (r["obs"] == m and r["group"] in modes_games))]
    if not any(pick("one", t, s, 1) for t in traits for s in (1, -1)):
        return out[0].split("<p class=muted>")[0] + ("<p class=muted>Not in this sweep: the comparison needs the "
                                                      "vector at c = ±1 (the prompt's matched strength).</p>")
    count = defaultdict(lambda: defaultdict(int))
    for t in traits:
        for s in (1, -1):
            v, p, b = pick("one", t, s, 1), pick("prompt_one", t, s), pick("both_one", t, s, 1)
            if not (v and p and b):
                continue
            lang, res = [], []
            for m in (None, "text", "image"):
                arms = [pick(c, t, s, at, m) for c, at in (("both_one", 1), ("one", 1), ("prompt_one", None))]
                mu, se, _ = R.random_effects([v_ for (cell, t_, s_, k_), v_ in per_game.items()
                                              if (t_, s_, k_) == (t, s, "readout_raw")
                                              and (m is None or (cell[1] == m and cell[0] in modes_games))])
                z = mu / se if math.isfinite(se) and se > 0 else float("nan")
                lang.append("<span class=muted>not measured</span>" if not MONITORED else
                            "<span class=muted>too little data</span>" if not math.isfinite(z) else
                            "more than the sum" if z > 2.58 else "less than the sum" if z < -2.58 else "adds up")
                rp = lambda cond, c=None: [r for r in paired_by.get(("*", t, cond), []) if r["steered"]
                                           and r["coef"] * s > 0 and (c is None or abs(r["coef"]) == c)
                                           and math.isfinite(r["diff"])
                                           and (m is None or (r["obs"] == m and r["group"] in modes_games))]
                res.append(_together([R._episode_mean([r["diff"] / SCALE.get(r["cell"], 1.0) for r in x],
                                                      [(r["cell"], r["episode"]) for r in x])
                                      for x in (rp("both_one", 1), rp("one", 1), rp("prompt_one"))]))
                count[("language", m)][lang[-1]] += 1
                count[("result", m)][res[-1]] += 1
            games = {}
            for kind in ("readout", "reward"):
                zs = [v_["b"] / v_["se"] for (cell, t_, s_, k_), v_ in per_game.items() if (t_, s_, k_) == (t, s, kind)]
                n_more, n_less = sum(z > 2.58 for z in zs), sum(z < -2.58 for z in zs)
                games[kind] = (f"<td>{n_more} / {len(zs) - n_more - n_less} / {n_less}</td>" if zs
                               else "<td class=muted>–</td>")
                count[("games", kind)]["more"] += n_more
                count[("games", kind)]["less"] += n_less
                count[("games", kind)]["all"] += len(zs)
            out.append(f"<tr><td class=l>{'more' if s > 0 else 'less'} {t}</td>"
                       + "".join(f"<td>{language(x, t)[0]}</td>" for x in (v, p, b))
                       + "".join(f"<td><b>{x}</b></td>" for x in lang) + games["readout"]
                       + "".join(f"<td>{x}</td>" for x in res) + games["reward"] + "</tr>")
    tally = lambda kind, m: ", ".join(f"{k} in {n}" for k, n in sorted(count[(kind, m)].items(), key=lambda kv: -kv[1])
                                      if "too little" not in k)
    all_g = sorted({r["cell"] for t in traits for r in by_trait.get(("prompt_one", t), [])})
    out.append("</table></div>" + R.games_note(all_g, "Pooled over all games and counted game by game")
               + R.games_note([(g, m) for g in sorted(modes_games) for m in ("text", "image")],
                              "Text and image columns (games played both ways)")
               + ("<p>Overall, on the language: " + tally("language", None)
                  + " of the persona × direction cases"
                  + (f" (with text: {tally('language', 'text')}; with images: {tally('language', 'image')})"
                     if modes_games else "") + ". " if MONITORED else "<p>")
               + "On the result: " + tally("result", None)
               + (f" (with text: {tally('result', 'text')}; with images: {tally('result', 'image')})"
                  if modes_games else "") + ".</p>"
               + "".join(f"<p>Game by game, on the {what}: more than the sum in {count[('games', k)]['more']}, less in "
                         f"{count[('games', k)]['less']} of {count[('games', k)]['all']} game × persona × direction "
                         "cases (with this many comparisons a few are chance).</p>"
                         for k, what in (("readout", "language"), ("reward", "result")) if count[("games", k)]["all"]))
    return "".join(out)


def text_vs_image(rows, traits, by, paired_by):
    """Games played from text and from pictures: how unsteered play differs, and whether steering's effect on the
    language and on the result is stronger or weaker with images, per setting."""
    base = R.mode_baselines(rows)
    if not base:
        return ""
    out = ["<h2 id=image>Text vs image: what changes when agents see the game as a picture?</h2>"
           "<h3>Without steering</h3><p class=muted>The same deals and seats in both modes (same seeds): each seat's "
           "result with images against its own result with text.</p><div class=wrap><table><tr><th class=l>game</th>"
           "<th>result, text → image</th><th>broken answers, text → image</th><th>answers, text → image</th>"
           "<th>tokens read per turn, text → image</th><th>turns per game, text → image</th></tr>"]
    for g, d in base.items():
        t, i = d["text"], d["image"]
        m, level = judge([x for _, x in d["paired"]], [k for k, _ in d["paired"]])
        res = (f"{t['win']:.0f}% → <b>{i['win']:.0f}%</b> wins" if t["binary"] else
               f"{number(t['reward'])} → <b>{number(i['reward'])}</b>")
        tok = (f"{t['tokens_in']:.0f} → {i['tokens_in']:.0f}" if math.isfinite(t["tokens_in"]) and
               math.isfinite(i["tokens_in"]) else "<span class=muted>not counted yet</span>")
        ratio = i["chars"] / t["chars"] if t["chars"] else float("nan")
        out.append(f"<tr><td class=l>{html.escape(g)}</td><td>{res} {mark(level)}</td>"
                   f"<td>{t['unparseable']:.0f}% → {i['unparseable']:.0f}%</td>"
                   f"<td>{'about as long' if 0.8 <= ratio <= 1.25 else f'{ratio:.1f}× as long'}</td><td>{tok}</td>"
                   f"<td>{t['length']:.0f} → {i['length']:.0f}</td></tr>")
    out.append("</table></div>")

    # steering, text vs image: at full strength, per game x persona x setting, language and result
    tally = {k: defaultdict(int) for k in ("language", "result")}
    for g in base:
        for t in traits:
            for c in SETTINGS:
                for coef in (-1.0, 1.0) if "prompt" in c else (-2.0, 2.0):
                    lang = [[r for r in by.get(((g, m), t, c), []) if r["steered"] and r["coef"] == coef
                             and t in r["mon"]] for m in ("text", "image")]
                    res = [[r for r in paired_by.get(((g, m), t, c), []) if r["steered"] and r["coef"] == coef
                            and math.isfinite(r["diff"])] for m in ("text", "image")]
                    for kind, sides, val, cl in (
                            ("language", lang, lambda r: math.copysign(1, coef) * r["mon"][t], lambda r: r["cluster"]),
                            ("result", res, lambda r: r["diff"], lambda r: (r["cell"], r["episode"]))):
                        if not sides[0] or not sides[1]:
                            continue
                        (mt, st, _), (mi, si, _) = (R._episode_mean([val(r) for r in x], [cl(r) for r in x])
                                                    for x in sides)
                        se = math.hypot(st, si)
                        if math.isfinite(se) and se > 0:
                            z = (mi - mt) / se
                            word = ("larger with images" if z > 2.58 else "smaller with images" if z < -2.58
                                    else "about the same") if kind == "result" else (
                                "stronger with images" if z > 2.58 else "weaker with images" if z < -2.58
                                else "about the same")
                            tally[kind][(c, word)] += 1
    out.append("<h3>With steering: is the effect stronger with images?</h3><p class=muted>Per setting, over every "
               "game played both ways × persona × direction at full strength (the vector at c = ±2, the prompt ±): "
               "how often the language moves further the intended way with images than with text (or less), and how "
               "often the result changes more (higher) or less with images. 'Tentative or solid' differences only. "
               "Each number counts game × persona × direction cases.</p>"
               "<div class=wrap><table><tr><th class=l>who is steered, and how</th>"
               "<th>language: moves further the intended way with images</th><th>language: moves less far with images"
               "</th><th>language: no clear difference</th><th>result: higher with images</th><th>result: lower with "
               "images</th><th>result: no clear difference</th></tr>")
    for c in SETTINGS:
        L = [tally["language"][(c, w)] for w in ("stronger with images", "weaker with images", "about the same")]
        Rr = [tally["result"][(c, w)] for w in ("larger with images", "smaller with images", "about the same")]
        if sum(L) or sum(Rr):
            out.append(f"<tr><td class=l>{html.escape(SETTING_WORDS[c])}</td>"
                       + "".join(f"<td>{n}</td>" if MONITORED else "<td class=muted>not measured</td>" for n in L)
                       + "".join(f"<td>{n}</td>" for n in Rr) + "</tr>")
    out.append("</table></div>" + R.games_note([(g, m) for g in base for m in ("text", "image")], "Games in this table (played both ways)"))

    # language over c, text vs image, per persona: games played both ways, the vector settings without a side
    mc = {"text": "var(--s1)", "image": "var(--s2)"}
    charts = []
    for t in traits:
        series = {m: curve([r for g in base for c in ("one", "half", "all") for r in by.get(((g, m), t, c), [])
                            if r["steered"]], lambda r: r["mon"].get(t)) for m in ("text", "image")}
        charts.append(f"<div class=chart><h3>{html.escape(t)}</h3>"
                      + R.line_chart(series, t, f"{t} in language", colours=mc) + "</div>")
    if MONITORED:
        out.append("<h3>The persona in the language over c, text vs image</h3><div class=legend><span><i style="
                   "'background:var(--s1)'></i>text</span><span><i style='background:var(--s2)'></i>image</span><span>"
                   "one agent, half or every agent with the vector, the games played both ways; 0 = like unsteered "
                   "agents; bars: 95% CI</span></div><div class=charts>" + "".join(charts) + "</div>"
                   + R.games_note([(g, m) for g in base for m in ("text", "image")], "Games in these charts"))

    # the result over c, text vs image, with the straight line through it: how fast the result changes with c
    charts = []
    for t in traits:
        series, fits = {}, {}
        for m in ("text", "image"):
            rs = [r for g in base for c in ("one", "half", "all") for r in paired_by.get(((g, m), t, c), [])
                  if r["steered"] and math.isfinite(r["diff"])]
            val = lambda r: r["diff"] / SCALE.get(r["cell"], 1.0)
            series[m] = curve(rs, val, lambda r: (r["cell"], r["episode"]))
            fits[m] = R._line_fit([r["coef"] for r in rs], [val(r) for r in rs])
        charts.append(f"<div class=chart><h3>{html.escape(t)}</h3>"
                      + R.line_chart(series, t, "result change (SD of the game)", colours=mc, fits=fits) + "</div>")
    out.append("<h3>The result over c, text vs image</h3><div class=legend><span><i style='background:var(--s1)'></i>"
               "text</span><span><i style='background:var(--s2)'></i>image</span><span>the steered agents' result against "
               "the same seats unsteered, in units of each game's unsteered spread so games can be pooled; dashed: the "
               "straight line through the points (its steepness = how fast the result changes with c; hover it)</span>"
               "</div><div class=charts>" + "".join(charts) + "</div>"
               + R.games_note([(g, m) for g in base for m in ("text", "image")], "Games in these charts"))

    # prompt, vector, both: how far each moves the language, with text and with images
    if not MONITORED:
        return "".join(out)
    rows_ = ["<h3>Prompt, vector or both, with text and with images</h3><p class=muted>One agent; the vector and both "
             "at c = ±1; the games played both ways. How far the persona shows in the language, per mode.</p>"
             "<div class=wrap><table><tr><th class=l rowspan=2>persona</th><th colspan=2>vector alone (c = ±1)</th>"
             "<th colspan=2>prompt alone</th><th colspan=2>prompt + vector (c = ±1)</th></tr><tr>"
             + "<th>with text</th><th>with images</th>" * 3 + "</tr>"]
    gain = defaultdict(list)
    for t in traits:
        for s in (1, -1):
            cells_ = []
            for c, at in (("one", 1.0), ("prompt_one", None), ("both_one", 1.0)):
                for m in ("text", "image"):
                    rs = [r for g in base for r in by.get(((g, m), t, c), []) if r["steered"] and r["coef"] * s > 0
                          and (at is None or abs(r["coef"]) == at)]
                    w, mean, level = language(rs, t)
                    cells_.append(f"<td>{w}</td>")
                    if level >= 2:
                        gain[(c, m)].append(s * mean)
            rows_.append(f"<tr><td class=l>{'more' if s > 0 else 'less'} {t}</td>{''.join(cells_)}</tr>")
    rows_.append("</table></div>" + R.games_note(sorted({r["cell"] for g in base for m in ("text", "image")
                                                          for t in traits for c in ("one", "prompt_one", "both_one")
                                                          for r in by.get(((g, m), t, c), [])})))
    avg = lambda c, m: float(np.mean(gain[(c, m)])) if gain[(c, m)] else float("nan")
    words_ = {"one": "the vector", "prompt_one": "the prompt", "both_one": "both together"}
    trend = [f"{words_[c]} {'move' if c == 'both_one' else 'moves'} it "
             f"{'further' if avg(c, 'image') > avg(c, 'text') else 'less far'} with images"
             for c in words_ if math.isfinite(avg(c, "text")) and math.isfinite(avg(c, "image"))]
    if trend:
        rows_.append(f"<p>Where the effect is clear, {'; '.join(trend)} (the intended way, on average over personas).</p>")
    out.append("".join(rows_) + "<p class=muted>Per game and setting, text and image results side by side: in the "
               "per-game charts below.</p>")
    return "".join(out)


def language_vs_action(rows, cells):
    """Per game: the clearest changes in what steered agents choose, by setting and strength."""
    out = ["<h2 id=actions>Language vs action: does the persona show in the moves?</h2><p class=muted>Per game: the "
           "choices that change most clearly with the persona at full strength (the vector at c = ±2, the prompt ±): "
           "the share of an action type, or the mean of a numeric choice (bid, guess, amount), against unsteered "
           "agents of the same game. Each line says who is steered.</p>"]
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell and r["act_n"]]
        base = [r for r in cr if r["condition"] == "none"]
        if len(base) < 3:
            continue
        total = defaultdict(float)
        for r in base:
            for lab, k in r["act_labels"].items():
                total[lab] += k
        n = sum(total.values()) or 1
        measures = [(f"'{lab}'", lambda r, lab=lab: 100 * r["act_labels"].get(lab, 0) / r["act_n"], "%")
                    for lab, k in total.items() if k / n >= 0.05 and lab not in ("text", "move", "number")]
        if sum(r["act_num"] is not None for r in base) >= len(base) / 2:
            q1, q3 = np.percentile([r["act_num"] for r in cr if r["act_num"] is not None], [25, 75])
            ok = lambda v: v is not None and q1 - 3 * (q3 - q1) <= v <= q3 + 3 * (q3 - q1)
            measures.append(("the numeric choice", lambda r: r["act_num"] if ok(r["act_num"]) else None, ""))
        steered = R._by([r for r in cr if r["steered"] and r["condition"] in SETTINGS and not r["random"]
                         and (abs(r["coef"] or 0) == 2 or r["via"] == "prompt")],
                        lambda r: (r["trait"], r["condition"], r["coef"]))
        found = []
        for (t, c, coef), st in steered.items():
            for name, val, unit in measures:
                b0 = [val(r) for r in base if val(r) is not None]
                sv = [(val(r), r["cluster"]) for r in st if val(r) is not None]
                if len(b0) < 3 or len(sv) < 5:
                    continue
                m0 = float(np.mean(b0))
                m, level = judge([v - m0 for v, _ in sv], [k for _, k in sv])
                if level >= 2:
                    strength = f"{'+' if coef > 0 else '−'} instruction" if "prompt" in c else f"c = {coef:+g}"
                    found.append((abs(m) / (float(np.std(b0)) or 1.0),
                                  f"{t}, {SETTING_WORDS.get(c, c)}, {strength}: {name} {number(m0 + m, unit)} "
                                  f"(unsteered {number(m0, unit)}) {mark(level)}"))
        if found:
            found.sort(key=lambda x: -x[0])
            out.append(f"<details><summary>{html.escape(R.cell_name(cell))}: {len(found)} clear changes in the moves"
                       f"</summary><ul>" + "".join(f"<li>{f}</li>" for _, f in found[:15]) + "</ul></details>")
    return "".join(out)


def roles(rows, traits):
    """Hidden-role games: the deceivers' win rate over c, one line per carrier."""
    wins = R.deceiver_wins(rows)
    cells = sorted({w["cell"] for w in wins})
    if not cells:
        return ""
    by = R._by(wins, lambda w: (w["cell"], w["condition"], w["trait"]))
    sets = [c for c in ("evil_team", "evil_one", "good_team", "good_one", "all", "one", "evil_team_prompt")
            if any(k[1] == c for k in by)]
    out = ["<h2 id=roles>Hidden roles: who carries the persona, and who wins</h2><p class=muted>The deceiving side's "
           "win rate (spies, mafia, werewolves) over the steering strength, one line per carrier; the unsteered rate "
           "at c = 0. Hover a line or a point.</p>", legend(sets + ["unsteered"])]
    for cell in cells:
        b0 = [w["won"] for w in wins if w["cell"] == cell and w["condition"] == "none"]
        base = R.mean_ci(b0, list(range(len(b0))))
        panels = []
        for t in traits:
            series = {c: curve(by[(cell, c, t)], lambda w: w["won"])[:-1] for c in sets if by.get((cell, c, t))}
            if b0:
                series["unsteered"] = [(0.0, base[0], base[1])]
            panels.append((t, series))
        out.append(f"<details><summary>{html.escape(R.cell_name(cell))}: deceivers win "
                   f"{np.mean(b0) if b0 else float('nan'):.0f}% unsteered (n = {len(b0)})</summary><div class=charts>"
                   + charts_row(panels, "deceivers win (%)") + "</div></details>")
    return "".join(out)


def gallery(rows, logs, k=6, chars=500):
    """Real answers where steering breaks the format (the game cannot parse them) or fixes it."""
    roots = [Path(d) for d in logs or [] if Path(d).is_dir()]
    none = R._by([r for r in rows if r["condition"] == "none" and r["turns"]], lambda r: r["cell"])
    none = {c: rs for c, rs in none.items() if len({r["episode"] for r in rs}) >= 5}  # a rate needs a few episodes
    rate0 = {c: sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs)) for c, rs in none.items()}
    cells = R._by([r for r in rows if r["steered"] and r["turns"] and r["coef"] and r["cell"] in rate0],
                  lambda r: (r["cell"], r["trait"], r["coef"], r["via"]))
    rate = {k_: sum(r["fallbacks"] for r in rs) / max(1, sum(r["turns"] for r in rs)) for k_, rs in cells.items()
            if len(rs) >= 20}
    broken = sorted((k_ for k_ in rate if rate[k_] - rate0[k_[0]] >= 0.25), key=lambda k_: -(rate[k_] - rate0[k_[0]]))
    fixed = sorted((k_ for k_ in rate if rate0[k_[0]] - rate[k_] >= 0.2), key=lambda k_: rate[k_] - rate0[k_[0]])

    def answer(r, want_ok):
        for root in roots:
            path = root / r["group"] / r["run_id"] / f"episode_{r['episode']}.jsonl"
            if path.exists():
                for rec in R._turns(path):
                    if rec.get("agent_id") == r["agent"] and bool(rec.get("parse_ok", True)) == want_ok:
                        text = html.escape((rec.get("completion") or "").strip()[:chars]).replace("\n", "<br>")
                        return f"<div class=quote>{text or '<i>(empty answer)</i>'}</div>"
        return ""

    def show(keys, want_ok):
        items = []
        for key in keys[:k]:
            cell, t, c, via = key
            r = max(cells[key], key=lambda r: (r["fallbacks"] if not want_ok else -r["fallbacks"]) / r["turns"])
            items.append(f"<li><b>{html.escape(R.cell_name(cell))}</b>, {t} {via}, c = {c:+g}: "
                         f"{100 * rate[key]:.0f}% of turns unparseable (unsteered {100 * rate0[cell]:.0f}%)"
                         + answer(r, want_ok) + "</li>")
        return "<ul>" + "".join(items) + "</ul>" if items else "<p class=muted>none</p>"
    note = "" if roots else "<p class=muted>(answers appear when the report is built with --logs)</p>"
    return ("<h2 id=gallery>Failure gallery: where steering breaks or fixes the answers</h2><p class=muted>A turn the "
            "game cannot parse even after the retries makes the agent play a default move, so in these games part "
            "of any result change is a format effect.</p>" + note
            + f"<h3>Breaks: steered answers the game could not parse</h3>{show(broken, False)}"
            + f"<h3>Fixes: steering makes answers parse where unsteered ones often do not</h3>{show(fixed, True)}")


def per_game(traits, by, paired_by, wins):
    """Per game and steering vector: language and result over c, one line per setting; text and image side by side."""
    out = ["<h2 id=games>Per game and steering vector: every setting</h2><p class=muted>Click a game. Per persona, "
           "line plots over the steering strength c, one line per setting (who is steered, and how): the steered "
           "agents' <b>language</b> (the persona in what they write; 0 = like unsteered agents) and their "
           "<b>result</b> (change against the same seats in the unsteered games with the same seeds: in percentage "
           "points of wins, or in the game's score). Charts of one kind share their axis; text and image side by "
           "side where the game was played both ways. Hover a line for the setting, a point for its value.</p>"]
    groups = defaultdict(list)
    for cell in sorted({k[0] for k in by}):
        groups[cell[0]].append(cell)
    for g, gcells in sorted(groups.items()):
        gcells.sort(key=lambda c: c[1] != "text")  # text first, then image
        sets = [c for c in SETTINGS if any((cell, t, c) in by for cell in gcells for t in traits)]
        blocks = []
        for t in traits:
            lang, res = [], []
            for cell in gcells:
                mode = "" if len(gcells) == 1 else f" ({cell[1]})"
                lang.append((f"language{mode}", {c: curve([r for r in by[(cell, t, c)] if r["steered"]],
                                                          lambda r: r["mon"].get(t)) for c in sets if (cell, t, c) in by}))
                k = 100.0 if cell in wins else 1.0
                res.append((f"result{mode}", {c: curve([r for r in paired_by.get((cell, t, c), []) if r["steered"]
                                                        and math.isfinite(r["diff"])], lambda r: k * r["diff"],
                                                       lambda r: (r["cell"], r["episode"]))
                                              for c in sets if paired_by.get((cell, t, c))}))
            unit = "change in wins (points)" if all(cell in wins for cell in gcells) else "change in score"
            blocks.append(f"<h3>{html.escape(t)}</h3><div class='charts four'>"
                          f"{charts_row(lang, f'{t} in language') if MONITORED else ''}{charts_row(res, unit)}</div>")
        out.append(f"<details><summary>{html.escape(g)}</summary>{legend(sets)}{''.join(blocks)}</details>")
    return "".join(out)


CSS_EXTRA = """
.charts.four { grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); }
.mk { font-size:11px; letter-spacing:-1px; color:var(--pos); white-space:nowrap; }
.quote { margin:4px 0 10px; padding:6px 10px; border-left:3px solid var(--line); font-size:13px; color:var(--muted); }
"""
SCALE = {}
MONITORED = True  # False for a sweep played without the monitor: no language results, only game results


def build(rows, logs=None) -> str:
    global MONITORED
    rows = R.zscored(rows)
    SCALE.update(R.reward_scale(rows))
    MONITORED = any(r["mon"] for r in rows)
    traits = sorted({r["trait"] for r in rows if r["trait"]})
    cells = sorted({r["cell"] for r in rows})
    base = R.baselines(rows)
    by = R._by([r for r in rows if r["trait"]], lambda r: (r["cell"], r["trait"], r["condition"]))
    by_trait = R._by([r for r in rows if r["trait"]], lambda r: (r["condition"], r["trait"]))
    pr = R.paired(rows, base, lambda r: r["condition"] != "none" and r["trait"] is not None)
    paired_by = R._by(pr, lambda r: (r["cell"], r["trait"], r["condition"]))
    for r in pr:  # pooled over games, for the synergy table
        paired_by.setdefault(("*", r["trait"], r["condition"]), []).append(r)
    last = R.last_step_only(rows)
    intro = ("<div class=box><b>What this is</b><span>Persona-vector steering of LLM agents in multi-agent games, in "
             f"plain terms: {len(cells)} games (game × text or image), {len(traits)} personas. "
             "The full report (report.html) has the statistics behind every line.</span><b>How to read it</b><span>"
             "Charts: the steering strength c on the x axis (negative = pushed away from the persona, 0 = unsteered), "
             "one line per setting. Words: markers "
             f"{MARK[3]} solid (p &lt; .001), {MARK[2]} tentative (p &lt; .01; with this many comparisons some are "
             f"chance), {MARK[1]} no clear change, {MARK[0]} too little data. Results are the game's own (wins, or the "
             "score in the game's units), always against the same seats in unsteered games with the same seeds."
             "</span><b>How the monitor reads language</b><span>"
             + (R.monitor_how(len({t for r in rows for t in r["mon"]}),
                              sorted({r["monitor_layer"] for r in rows if r.get("monitor_layer") is not None}))
                if MONITORED else "<b>Not run for these episodes</b> (played without the monitor engine): every "
                "'language' result reads 'not measured' until the monitor is run over the logged turns; the game "
                "results, unparseable turns, answer length and moves do not need it.")
             + "</span></div>")
    warn = (f"<div class=warn><b>Provisional results for {html.escape(', '.join(last))}:</b> these games pay every "
            "step, but this data has only their last step's payoff; their results change once the report is rebuilt "
            "with episode totals.</div>" if last else "")
    toc = ("<p class=muted>Contents: <a href=#headline>headline</a> · <a href=#recipe>recipe</a> · "
           "<a href=#synergy>prompt, vector, both</a> · <a href=#image>text vs image</a> · <a href=#actions>language vs "
           "action</a> · <a href=#roles>hidden roles</a> · <a href=#gallery>failure gallery</a> · "
           "<a href=#examples>examples</a> · <a href=#games>per game</a></p>")
    coefs = sorted({r["coef"] for r in rows if r["coef"] is not None and r["via"] != "prompt"})
    parts = [f"<h2 id=headline>Headline</h2><div class=result><ul>"
             + "".join(f"<li>{h}</li>" for h in headline(by, paired_by)) + "</ul></div>",
             recipe(rows, traits, by_trait, paired_by, cells),
             synergy(traits, by_trait, paired_by, {g for g in R.mode_baselines(rows)},
                     R.synergy_cells(rows, traits, base, SCALE)),
             text_vs_image(rows, traits, by, paired_by), language_vs_action(rows, cells),
             roles(rows, traits),
             gallery(rows, logs), R.examples_section(logs, traits, coefs) if logs else "",
             per_game(traits, by, paired_by, win_games(rows))]
    R.CELL_NAMES.clear()
    R.CELL_NAMES.update(R.cell_name(c) for c in cells)

    def with_games(part):
        """The part with, right under its heading, the games its results come from."""
        games = R.section_games(part)
        if not games or "</h2>" not in part:
            return part
        head, rest = part.split("</h2>", 1)
        return (head + "</h2><p class=games><b>Games played for these results (" + str(len(games)) + "):</b> "
                + html.escape(", ".join(games)) + "</p>" + rest)
    parts = [with_games(x) for x in parts]
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width, "
            f"initial-scale=1'><title>Steering mini report</title><style>{R.CSS}{CSS_EXTRA}</style></head><body>"
            f"<h1>Steering mini report</h1>{intro}{warn}{toc}{''.join(parts)}</body></html>")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("agents_csv", help="output of scripts/summarize_runs.py")
    p.add_argument("--out", required=True)
    p.add_argument("--logs", nargs="+", help="the sweep's log dir(s): adds examples and the failure gallery's answers")
    args = p.parse_args(argv)
    rows = R.load(args.agents_csv)
    if not rows:
        print("no rows")
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(build(rows, logs=args.logs), encoding="utf-8")
    print(f"steering mini report over {len(rows)} agent-episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

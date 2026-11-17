"""How well the accuracy-debate agents do (qa_debate, config/sweeps/debate_acc.yaml), from the step logs.

    python scripts/debate_accuracy.py <log dir>/debate_acc [--by-n] [--out FILE]

1. Unsteered, per game group (dataset x N): agent accuracy in round 1 and in the final round, the group's majority
   accuracy, answer switches between round 1 and the final round (right->wrong, wrong->right, per agent), answers
   that could not be parsed (fallback) and answers cut off at the token limit.
2. Steered vs unsteered on the same questions (episode i = question i in every run): final majority accuracy and
   agent accuracy minus unsteered, mean +- SE over questions; pooled over N per dataset (--by-n: per group too).
   For 'one'-agent conditions, the steered agent's own accuracy is shown next to the other agents'.
Reads only the logs; changes nothing.
"""
import argparse
import collections
import json
import math
import re
from pathlib import Path

RUN = re.compile(r"^(?P<group>.+?)__(?P<cond>[a-z_]+?)(?:__(?P<trait>[a-z]+)_x(?P<c>-?\d+(?:\.\d+)?))?$")


def episode(path: Path):
    """Per-turn info and per-line flags of one finished episode, or None if it is unfinished."""
    if not path.with_suffix(".summary.json").exists():
        return None
    turns, steered, lines, fallback, cut = {}, set(), 0, 0, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a damaged line (e.g. NUL bytes left by the file system): skipped, counted
                BAD[str(path)] += 1
                continue
            lines += 1
            fallback += not r.get("parse_ok", True)
            cut += r.get("finish_reason") == "length"
            if r.get("steering"):
                steered.add(r["agent_id"])
            turns.setdefault(r["turn"], r.get("info") or {})
    if not turns:
        return None
    first, last = turns[min(turns)], turns[max(turns)]
    c1, cf = first.get("correct", {}), last.get("correct", {})
    agents = sorted(cf)
    return {"r1": mean([c1.get(a, False) for a in agents]), "final": mean([cf[a] for a in agents]),
            "maj": float(last.get("majority_correct", 0.0)), "maj_r1": float(first.get("majority_correct", 0.0)),
            "rw": sum(c1.get(a, False) and not cf[a] for a in agents) / len(agents),
            "wr": sum(not c1.get(a, False) and cf[a] for a in agents) / len(agents),
            "steered_acc": mean([cf[a] for a in agents if a in steered]) if steered else None,
            "other_acc": mean([cf[a] for a in agents if a not in steered]) if 0 < len(steered) < len(agents) else None,
            "fallback": fallback / lines, "cut": cut / lines}


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--by-n", action="store_true", help="also every steered cell per group, not only pooled over N")
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = collections.defaultdict(dict)  # (group, cond, trait, c) -> {episode: metrics}
    for run_dir in sorted(p for g in Path(a.logs).iterdir() if g.is_dir() for p in g.iterdir() if p.is_dir()):
        m = RUN.match(run_dir.name)
        if not m:
            continue
        key = (m["group"], m["cond"], m["trait"], float(m["c"]) if m["c"] else None)
        for f in run_dir.glob("episode_*.jsonl"):
            ep = episode(f)
            if ep is not None:
                runs[key][int(f.stem.split("_")[1])] = ep
    out = []
    say = out.append

    def nkey(g):
        ds, n = g.rsplit("_", 1)
        return ds, int(n[:-1])

    groups = sorted({k[0] for k in runs}, key=nkey)
    say("1. Unsteered (mean over questions; switches and fallbacks per agent answer)")
    say(f"{'group':18}{'eps':>5}{'acc r1':>8}{'acc fin':>9}{'maj r1':>8}{'maj fin':>9}{'R->W':>7}{'W->R':>7}"
        f"{'fallbk':>8}{'cut':>6}")
    base = {}
    for g in groups:
        eps = runs.get((g, "none", None, None), {})
        base[g] = eps
        if not eps:
            say(f"{g:18}    0")
            continue
        v = list(eps.values())
        say(f"{g:18}{len(v):>5}" + "".join(f"{mean([e[k] for e in v]):>{w}.3f}" for k, w in
                                           (("r1", 8), ("final", 9), ("maj_r1", 8), ("maj", 9), ("rw", 7), ("wr", 7))) +
            f"{100 * mean([e['fallback'] for e in v]):>7.1f}%{100 * mean([e['cut'] for e in v]):>5.1f}%")

    def rows(cells, label):
        say("")
        say(label)
        say(f"{'cell':44}{'qs':>5}{'d maj':>8}{'+-SE':>7}{'d acc':>8}{'+-SE':>7}{'steered':>9}{'others':>8}{'fallbk':>8}")
        for name, pairs in cells:
            if not pairs:
                continue
            dm = [s["maj"] - b["maj"] for s, b in pairs]
            da = [s["final"] - b["final"] for s, b in pairs]
            say(f"{name:44}{len(pairs):>5}{mean(dm):>+8.3f}{se(dm):>7.3f}{mean(da):>+8.3f}{se(da):>7.3f}"
                f"{mean([s['steered_acc'] for s, _ in pairs]):>9.3f}{mean([s['other_acc'] for s, _ in pairs]):>8.3f}"
                f"{100 * mean([s['fallback'] for s, _ in pairs]):>7.1f}%")

    def paired(g, cond, trait, c):
        eps, b = runs.get((g, cond, trait, c), {}), base.get(g, {})
        return [(eps[i], b[i]) for i in sorted(eps) if i in b]

    steered = sorted({k[1:] for k in runs if k[1] != "none"}, key=lambda k: (k[0], k[1], k[2]))
    for ds in sorted({nkey(g)[0] for g in groups}):
        gs = [g for g in groups if nkey(g)[0] == ds]
        rows([(f"{cond} {trait} c={c:+g}", [p for g in gs for p in paired(g, cond, trait, c)])
              for cond, trait, c in steered],
             f"2. {ds}, every N pooled: steered minus unsteered on the same questions")
        if a.by_n:
            for g in gs:
                rows([(f"{cond} {trait} c={c:+g}", paired(g, cond, trait, c)) for cond, trait, c in steered],
                     f"   {g}: steered minus unsteered")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

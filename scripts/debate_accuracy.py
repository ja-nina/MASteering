"""How well the accuracy-debate agents do (qa_debate, config/sweeps/debate_acc.yaml), from the step logs.

    python scripts/debate_accuracy.py <log dir>/debate_acc [--by-n] [--out FILE] [--extra-baseline <log dir>/OTHER]

1. Unsteered, per game group (dataset x N): agent accuracy in round 1 and in the final round, the group's majority
   accuracy, answer switches between round 1 and the final round (right->wrong, wrong->right, per agent), answers
   that could not be parsed (fallback) and answers cut off at the token limit.
2. Steered vs unsteered on the same questions (episode i = question i in every run): final majority accuracy and
   agent accuracy minus unsteered, mean +- SE over questions; pooled over N per dataset (--by-n: per group too).
   Pooled cells meet each question once per N; those copies are not independent, so each question's differences
   are averaged over N first and the mean and SE are over distinct questions ('qs' counts them).
   For 'one'-agent conditions, the steered agent's own accuracy is shown next to the other agents'.
   Every cell is compared with the same unsteered run, so that run's sampling luck shifts all cells alike (the SE
   does not include it). Two remedies: --extra-baseline adds the unsteered runs of another sweep with the same game
   groups and questions (e.g. a pilot) as more samples per question, averaged; and for 'all' cells, 'd rnd' is the
   majority accuracy minus that of all_random at the same trait and coefficient (a random direction of the same
   size, on the same questions), which needs no unsteered baseline at all.
Reads only the logs; changes nothing.
"""
import argparse
import collections
import json
import math
import re
from pathlib import Path

BAD = collections.Counter()  # episode log -> damaged lines skipped
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


def load(logs):
    """(group, cond, trait, c) -> {episode: metrics} of every finished episode under a sweep's log dir."""
    runs = collections.defaultdict(dict)
    for run_dir in sorted(p for g in Path(logs).iterdir() if g.is_dir() for p in g.iterdir() if p.is_dir()):
        m = RUN.match(run_dir.name)
        if not m:
            continue
        key = (m["group"], m["cond"], m["trait"], float(m["c"]) if m["c"] else None)
        for f in run_dir.glob("episode_*.jsonl"):
            ep = episode(f)
            if ep is not None:
                runs[key][int(f.stem.split("_")[1])] = ep
    return runs


def average(samples):
    """One question's metrics averaged over several unsteered samples of it."""
    return {k: (mean([s[k] for s in samples]) if samples[0][k] is not None else None) for k in samples[0]}


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def by_question(diffs):
    """Mean +- SE of (question, difference) pairs with the question as the unit: a question met at several N
    (pooled cells) is averaged over them first, so its N copies count once and the SE is over distinct questions."""
    per_q = collections.defaultdict(list)
    for i, d in diffs:
        per_q[i].append(d)
    qs = [mean(v) for v in per_q.values()]
    return mean(qs), se(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--by-n", action="store_true", help="also every steered cell per group, not only pooled over N")
    ap.add_argument("--out")
    ap.add_argument("--extra-baseline", action="append", default=[],
                    help="another sweep's log dir: its unsteered runs (same groups, same questions) join the baseline")
    a = ap.parse_args()
    if not Path(a.logs).is_dir():
        raise SystemExit(f"no logs at {a.logs} yet: no episode of this sweep has started (squeue / sacct)")
    runs = load(a.logs)
    extra = collections.defaultdict(lambda: collections.defaultdict(list))  # group -> episode -> [metrics]
    for d in a.extra_baseline:
        if not Path(d).is_dir():
            raise SystemExit(f"no logs at {d}")
        for (g, cond, _, _), eps in load(d).items():
            if cond == "none":
                for i, e in eps.items():
                    extra[g][i].append(e)
    out = []
    say = out.append

    def nkey(g):
        ds, n = g.rsplit("_", 1)
        return ds, int(n[:-1])

    groups = sorted({k[0] for k in runs}, key=nkey)
    say("1. Unsteered (mean over questions; switches and fallbacks per agent answer)")
    say(f"{'group':18}{'eps':>5}{'acc r1':>8}{'acc fin':>9}{'maj r1':>8}{'maj fin':>9}{'R->W':>7}{'W->R':>7}"
        f"{'fallbk':>8}{'cut':>6}")
    base, n_base = {}, collections.Counter()
    for g in groups:
        eps = runs.get((g, "none", None, None), {})
        base[g] = {}
        for i in set(eps) | set(extra[g]):
            samples = ([eps[i]] if i in eps else []) + extra[g][i]
            base[g][i] = average(samples)
            n_base[len(samples)] += 1
        if not eps:
            say(f"{g:18}    0")
            continue
        v = list(eps.values())
        say(f"{g:18}{len(v):>5}" + "".join(f"{mean([e[k] for e in v]):>{w}.3f}" for k, w in
                                           (("r1", 8), ("final", 9), ("maj_r1", 8), ("maj", 9), ("rw", 7), ("wr", 7))) +
            f"{100 * mean([e['fallback'] for e in v]):>7.1f}%{100 * mean([e['cut'] for e in v]):>5.1f}%")

    if a.extra_baseline:
        say("")
        say("Baseline for section 2: unsteered samples per question (this sweep + --extra-baseline): "
            + ", ".join(f"{k} sample(s): {v} question(s)" for k, v in sorted(n_base.items())))

    def rows(cells, label):
        say("")
        say(label)
        say(f"{'cell':44}{'qs':>5}{'d maj':>8}{'+-SE':>7}{'d acc':>8}{'+-SE':>7}{'steered':>9}{'others':>8}{'fallbk':>8}"
            f"{'d rnd':>8}{'+-SE':>7}")
        for name, pairs, rnd in cells:
            if not pairs:
                continue
            dm, sm = by_question([(i, s["maj"] - b["maj"]) for i, s, b in pairs])
            da, sa = by_question([(i, s["final"] - b["final"]) for i, s, b in pairs])
            dr, sr = by_question([(i, s["maj"] - r["maj"]) for i, s, r in rnd])
            say(f"{name:44}{len({i for i, _, _ in pairs}):>5}{dm:>+8.3f}{sm:>7.3f}{da:>+8.3f}{sa:>7.3f}"
                f"{mean([s['steered_acc'] for _, s, _ in pairs]):>9.3f}"
                f"{mean([s['other_acc'] for _, s, _ in pairs]):>8.3f}"
                f"{100 * mean([s['fallback'] for _, s, _ in pairs]):>7.1f}%"
                + (f"{dr:>+8.3f}{sr:>7.3f}" if rnd else ""))

    def paired(g, cond, trait, c):
        eps, b = runs.get((g, cond, trait, c), {}), base.get(g, {})
        return [(i, eps[i], b[i]) for i in sorted(eps) if i in b]

    def vs_random(g, cond, trait, c):  # 'all' only: all_random is its control (every agent, same size)
        if cond != "all":
            return []
        eps, r = runs.get((g, cond, trait, c), {}), runs.get((g, "all_random", trait, c), {})
        return [(i, eps[i], r[i]) for i in sorted(eps) if i in r]

    steered = sorted({k[1:] for k in runs if k[1] != "none"}, key=lambda k: (k[0], k[1], k[2]))
    for ds in sorted({nkey(g)[0] for g in groups}):
        gs = [g for g in groups if nkey(g)[0] == ds]
        rows([(f"{cond} {trait} c={c:+g}", [p for g in gs for p in paired(g, cond, trait, c)],
               [p for g in gs for p in vs_random(g, cond, trait, c)]) for cond, trait, c in steered],
             f"2. {ds}, every N pooled: steered minus unsteered on the same questions")
        if a.by_n:
            for g in gs:
                rows([(f"{cond} {trait} c={c:+g}", paired(g, cond, trait, c), vs_random(g, cond, trait, c))
                      for cond, trait, c in steered],
                     f"   {g}: steered minus unsteered")
    if BAD:
        say("")
        say(f"Skipped {sum(BAD.values())} damaged log line(s) in {len(BAD)} episode file(s), e.g. {next(iter(BAD))}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

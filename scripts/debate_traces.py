"""What a steered debate agent writes differently from an unsteered one, on the same questions (qa_debate logs).

    python scripts/debate_traces.py <log dir>/debate_acc_qwen3.5-2b --group qa_debate_hard_1p \
        --cell all__assertive_x1 --cell all_random__assertive_x1 [--examples 4] [--out FILE]

Compares each --cell run with the group's unsteered run (<group>__none), episode by episode (episode i = question i):
1. Per run, mean per agent turn: accuracy (final round), output tokens, turns cut at the token limit, unparseable
   turns, answer changes between rounds, 'wait'/'hmm'-style self-corrections and repeated lines per 1000 words,
   share of non-ASCII characters (language drift).
2. Examples: questions the unsteered agent got right and the first --cell got wrong, with both agents' answer per
   round and the start and end of their round-1 output, so the reasoning can be read side by side.
Reads only the logs; changes nothing. With N = 1 (a lone agent) any difference is in how the agent reasons, not in
how it debates.
"""
import argparse
import collections
import json
import re
from pathlib import Path

HEDGE = re.compile(r"\b(wait|hmm|actually|let me re-?(?:check|think|consider|read)|on second thought)\b", re.I)


def load(run_dir: Path):
    """{episode: [step records]} of the finished episodes of one run (damaged lines skipped)."""
    eps = {}
    for f in sorted(run_dir.glob("episode_*.jsonl")):
        if not f.with_suffix(".summary.json").exists():
            continue
        steps = []
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if steps:
            eps[int(f.stem.split("_")[1])] = steps
    return eps


def final_correct(steps):
    info = max(steps, key=lambda r: r["turn"]).get("info") or {}
    c = info.get("correct", {})
    return sum(map(bool, c.values())) / len(c) if c else 0.0


def answers(steps):
    """{agent: [normalized answer per round]} from the env's round info (the parsed action if it is missing)."""
    out = collections.defaultdict(list)
    for r in sorted(steps, key=lambda r: r["turn"]):
        got = (r.get("info") or {}).get("answers")
        out[r["agent_id"]].append(got.get(r["agent_id"]) if isinstance(got, dict) else r.get("parsed_action"))
    return out


def turn_stats(r):
    text = r.get("completion") or ""
    words = max(len(text.split()), 1)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    repeats = sum(n - 1 for n in collections.Counter(lines).values() if n > 1)
    return {"tokens": r.get("n_tokens") or r.get("tokens_out") or 0,
            "cut": r.get("finish_reason") == "length", "unparsed": not r.get("parse_ok", True),
            "hedges": 1000 * len(HEDGE.findall(text)) / words, "repeats": 1000 * repeats / words,
            "nonascii": sum(ord(ch) > 127 for ch in text) / max(len(text), 1)}


def summary(eps):
    turns = [turn_stats(r) for steps in eps.values() for r in steps]
    changes = [sum(a != b for a, b in zip(seq, seq[1:])) for steps in eps.values() for seq in answers(steps).values()]
    m = lambda xs: sum(map(float, xs)) / len(xs) if xs else float("nan")
    return {"eps": len(eps), "acc": m([final_correct(s) for s in eps.values()]),
            "tokens": m([t["tokens"] for t in turns]), "cut": m([t["cut"] for t in turns]),
            "unparsed": m([t["unparsed"] for t in turns]), "changes": m(changes),
            "hedges": m([t["hedges"] for t in turns]), "repeats": m([t["repeats"] for t in turns]),
            "nonascii": m([t["nonascii"] for t in turns])}


def clip(text, head=1200, tail=600):
    text = text or ""
    return text if len(text) <= head + tail else f"{text[:head]}\n   [... {len(text) - head - tail} chars ...]\n{text[-tail:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--group", required=True, help="e.g. qa_debate_hard_1p")
    ap.add_argument("--cell", action="append", required=True, help="run suffix after '<group>__', e.g. all__sycophantic_x1")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    gdir = Path(a.logs) / a.group
    runs = {"none": load(gdir / f"{a.group}__none")}
    for c in a.cell:
        if not (gdir / f"{a.group}__{c}").is_dir():
            raise SystemExit(f"no run {gdir / f'{a.group}__{c}'}")
        runs[c] = load(gdir / f"{a.group}__{c}")
    if not runs["none"]:
        raise SystemExit(f"no finished unsteered episodes in {gdir / f'{a.group}__none'}")
    common = set.intersection(*(set(e) for e in runs.values()))
    out = []
    say = out.append
    say(f"1. {a.group}, {len(common)} questions every run finished (mean per agent turn; per 1000 words for "
        f"self-corrections and repeated lines)")
    say(f"{'run':34}{'acc':>7}{'tokens':>8}{'cut':>7}{'unpars':>8}{'changes':>9}{'self-corr':>10}{'repeats':>9}{'non-ascii':>10}")
    for name, eps in runs.items():
        s = summary({i: eps[i] for i in common})
        say(f"{name:34}{s['acc']:>7.3f}{s['tokens']:>8.0f}{100 * s['cut']:>6.1f}%{100 * s['unparsed']:>7.1f}%"
            f"{s['changes']:>9.2f}{s['hedges']:>10.1f}{s['repeats']:>9.1f}{100 * s['nonascii']:>9.1f}%")

    first = a.cell[0]
    flips = [i for i in sorted(common) if final_correct(runs["none"][i]) > final_correct(runs[first][i])]
    say("")
    say(f"2. Questions unsteered got right and {first} got wrong: {len(flips)} of {len(common)}; "
        f"the first {min(a.examples, len(flips))}:")
    for i in flips[:a.examples]:
        say("")
        say("=" * 100)
        r0 = min(runs["none"][i], key=lambda r: r["turn"])
        info = r0.get("info") or {}
        say(f"episode {i}   correct answer: {info.get('correct_answer', '?')}   category: {info.get('category', '?')}")
        say("QUESTION (first turn's prompt, clipped):")
        say(clip(r0.get("user_prompt"), 900, 300))
        for name in ("none", first):
            steps = runs[name][i]
            say("-" * 100)
            say(f"[{name}]  answers per round: {dict(answers(steps))}   final correct: {final_correct(steps):.2f}")
            r1 = min(steps, key=lambda r: r["turn"])
            st = turn_stats(r1)
            say(f"round 1 of {r1['agent_id']}: {st['tokens']} tokens, finish={r1.get('finish_reason')}")
            say(clip(r1.get("completion")))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

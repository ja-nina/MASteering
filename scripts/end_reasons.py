"""How each game's episodes end: the game's own end reason (TextArena final_info reason, per-player close_info
reasons), the turn-limit cut, and whether an invalid move ended it, tallied per game group and observation.

    python scripts/end_reasons.py <log dir> [<fallback log dir> ...] [--setting none] [--max-c 1] [--out FILE]

Reads only episode_*.summary.json (and run.json for the group); logs nothing, changes nothing. Reasons are
normalised (player ids and numbers become #) so that one rule gives one row.
"""
import argparse
import collections
import json
import re
from pathlib import Path


def norm(text) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"[Pp]layer[ _]?\d+", "Player #", text)
    text = re.sub(r"\d+(\.\d+)?", "#", text)
    return text[:160]


def reasons(summary: dict) -> list:
    info = summary.get("final_info") or {}
    out = []
    if info.get("reason"):
        out.append(norm(info["reason"]))
    close = info.get("close_info")
    if isinstance(close, dict):
        if close.get("reason"):
            out.append(norm(close["reason"]))
        for pid, v in close.items():
            if isinstance(v, dict):
                if v.get("invalid_move"):
                    out.append("[invalid move] " + norm(v.get("reason", "")))
                elif v.get("reason"):
                    out.append(norm(v["reason"]))
    if summary.get("truncated"):
        out.append("[turn limit]")
    return sorted(set(out)) or ["[no reason logged]"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--setting", default=None, help="only runs with this setting (run id's 2nd field), e.g. none")
    ap.add_argument("--max-c", type=float, default=1.0, help="only |c| <= this (unsteered runs always kept)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    count = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    seen = set()
    for root in map(Path, a.logs):
        for run_json in sorted(root.rglob("run.json")):
            run = json.loads(run_json.read_text(encoding="utf-8"))
            cfg = run.get("config", run)
            run_id = str(cfg.get("run_id") or run_json.parent.name)
            setting = run_id.split("__")[1] if "__" in run_id else ""
            if a.setting and setting != a.setting:
                continue
            m = re.search(r"_x(-?\d+(?:\.\d+)?)", run_id)
            if setting != "none" and m and abs(float(m.group(1))) > a.max_c + 1e-9:
                continue
            group = run_id.split("__")[0]
            obs = "image" if "__obs-image" in run_id else "text"
            key = f"{group} ({obs})"
            for s in sorted(run_json.parent.glob("episode_*.summary.json")):
                ep = (run_id, s.name)
                if ep in seen:  # the fallback dir holds the same episode
                    continue
                seen.add(ep)
                try:
                    summary = json.loads(s.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                total[key] += 1
                for r in reasons(summary):
                    count[key][r] += 1
    lines = []
    for key in sorted(count):
        lines.append(f"== {key}: {total[key]} episodes")
        for r, n in count[key].most_common(12):
            lines.append(f"  {100 * n / total[key]:5.1f}%  {r}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

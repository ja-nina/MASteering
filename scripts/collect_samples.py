"""Collect samples from every run of a sweep into one self-contained HTML page (+ a status CSV).

    python scripts/collect_samples.py logs/smoke --configs config/runs/smoke --out results/smoke_samples.html

Per run: status (ok / failed / not run: episodes with a summary vs. expected), game, players,
observation mode, steering (with the persona prompt), the episode outcome, and the first
--turns turns of episode 0 plus the first turn of every steered agent: system and user prompt,
images (embedded), completion, parsed action and the top monitor traits. Images are embedded,
so the page can be copied off the cluster and opened anywhere.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import yaml

CSS = """body{font-family:system-ui,sans-serif;margin:16px;max-width:1100px;background:#fafaf7;color:#222}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px;text-align:left}
.ok{color:#1a7f37}.failed{color:#c62828;font-weight:bold}.missing{color:#a15c00}
details{margin:4px 0}pre{white-space:pre-wrap;background:#f0eee8;padding:6px;font-size:12px;margin:2px 0}
.turn{border-left:4px solid #bbb;padding-left:8px;margin:10px 0}.steered{border-left-color:#c62828}
img{max-width:420px;border:1px solid #ccc;margin:2px}h2{margin-top:36px;border-top:2px solid #ddd;padding-top:10px}"""


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []


def run_status(run_dir: Path, expected: int) -> tuple:
    done = len(list(run_dir.glob("episode_*.summary.json")))
    started = len(list(run_dir.glob("episode_*.jsonl")))
    status = "ok" if done >= expected else "failed" if started > done else "missing"
    return status, done, started


def _turn_html(run_dir: Path, rec: dict, max_chars: int) -> str:
    steer = rec.get("steering")
    images = rec.get("images")
    img_html = ""
    if isinstance(images, list):
        for rel in images:
            path = run_dir / rel
            if path.exists():
                data = base64.b64encode(path.read_bytes()).decode()
                img_html += f'<img src="data:image/png;base64,{data}">'
    elif images:
        img_html = f"<i>{images} image(s), not saved (save_images: false)</i>"
    monitor = rec.get("monitor") or {}
    mon = ""
    if monitor:
        layer = sorted(monitor, key=int)[len(monitor) // 2]
        top = sorted(monitor[layer].items(), key=lambda kv: kv[1], reverse=True)[:3]
        mon = f"monitor L{layer}: " + ", ".join(f"{t}={v:.2f}" for t, v in top)

    rejected = "".join(
        f'<details><summary class=failed>attempt {a["attempt"] + 1} rejected: {html.escape(str(a["parse_error"]))}'
        f'</summary><pre>{html.escape(a["completion"] or "")}</pre></details>'
        for a in rec.get("attempts", [])[:-1])  # every attempt but the final one was rejected

    def clip(text):
        text = text or ""
        return html.escape(text if len(text) <= max_chars else text[:max_chars] + f"\n[... {len(text) - max_chars} more chars]")

    return (f'<div class="turn{" steered" if steer else ""}"><b>turn {rec["turn"]} — {html.escape(str(rec["agent_id"]))}'
            f'</b> steering: <b>{html.escape(steer["label"]) if steer else "none"}</b> · action: '
            f'<code>{html.escape(str(rec.get("parsed_action")))}</code> · retries {rec.get("parse_retries")}'
            f'{" · <span class=failed>FALLBACK</span>" if not rec.get("parse_ok", True) else ""}'
            f'{" · TRUNCATED" if rec.get("finish_reason") == "length" else ""} · {html.escape(mon)}'
            f'<details><summary>system prompt</summary><pre>{clip(rec.get("system_prompt"))}</pre></details>'
            f'<details><summary>user prompt</summary><pre>{clip(rec.get("user_prompt"))}</pre></details>'
            f'{rejected}'
            f'<details><summary>exact model input</summary><pre>{clip(rec.get("model_input"))}</pre></details>'
            f'{img_html}<pre>{clip(rec.get("completion"))}</pre></div>')


def run_section(run_dir: Path, run: dict, status: tuple, turns: int, max_chars: int) -> str:
    out = [f'<h2 id="{html.escape(run["run_id"])}">{html.escape(run["run_id"])} '
           f'<span class="{status[0]}">[{status[0]}]</span></h2>',
           f'<p>game <b>{html.escape(str(run["game"]["id"]))}</b> ({run["game"]["family"]}), '
           f'{run["game"]["players"]} players, observation <b>{run.get("observation", "text")}</b>, '
           f'episodes done {status[1]}/{run.get("episodes", 1)}</p>']
    for s in run.get("steering_resolved") or []:
        out.append(f'<p>steering: <b>{html.escape(s["label"])}</b>'
                   + (f'<pre>persona prompt: {html.escape(s["persona"])}</pre>' if s.get("persona") else "") + "</p>")
    summary_path = run_dir / "episode_0.summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        steered = ", ".join(f"{a}: {s['label']}" for a, s in summary.get("steered", {}).items()) or "nobody"
        out.append(f'<p>episode 0: {summary.get("turns")} turns{" (truncated)" if summary.get("truncated") else ""}; '
                   f'rewards <code>{html.escape(json.dumps(summary.get("final_rewards")))}</code>; '
                   f'steered <code>{html.escape(steered)}</code></p>')
    records = _read_jsonl(run_dir / "episode_0.jsonl")
    if records:
        fallbacks = sum(not r.get("parse_ok", True) for r in records)
        out.append(f"<p>{len(records)} logged turns, {fallbacks} parse fallback(s)</p>")
    chosen = records[:turns]
    for agent in {r["agent_id"] for r in records if r.get("steering")}:
        first = next(r for r in records if r["agent_id"] == agent)
        if first not in chosen:
            chosen.append(first)
    out += [_turn_html(run_dir, r, max_chars) for r in chosen]
    return "\n".join(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", help="logging dir of the sweep, e.g. logs/smoke")
    p.add_argument("--configs", help="generated config dir: runs that never started are listed as 'missing'")
    p.add_argument("--out", required=True)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-chars", type=int, default=4000)
    args = p.parse_args(argv)

    runs = {}  # run_id -> (run dir, run dict)
    for run_json in sorted(Path(args.logs).rglob("run.json")):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        runs[run["run_id"]] = (run_json.parent, run)
    if args.configs:
        for path in sorted(Path(args.configs).rglob("*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            runs.setdefault(cfg["run_id"], (Path(cfg.get("logging_dir", "logs")) / cfg["run_id"], cfg))

    rows, sections = [], []
    for run_id, (run_dir, run) in sorted(runs.items()):
        status = run_status(run_dir, run.get("episodes", 1))
        rows.append({"run_id": run_id, "game": run["game"]["id"], "players": run["game"]["players"],
                     "observation": run.get("observation", "text"),
                     "steering": "; ".join(s["label"] for s in run.get("steering_resolved") or []) or
                                 ("none" if not run.get("steering") else "(not resolved)"),
                     "status": status[0], "episodes_done": status[1], "episodes_started": status[2]})
        sections.append(run_section(run_dir, run, status, args.turns, args.max_chars))

    table = "".join(f'<tr><td><a href="#{html.escape(r["run_id"])}">{html.escape(r["run_id"])}</a></td>'
                    f'<td>{html.escape(str(r["game"]))}</td><td>{r["observation"]}</td>'
                    f'<td>{html.escape(r["steering"])}</td><td class="{r["status"]}">{r["status"]}</td>'
                    f'<td>{r["episodes_done"]}</td></tr>' for r in rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "failed", "missing")}
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Game samples</title><style>{CSS}</style></head>"
            f"<body><h1>Samples: {html.escape(args.logs)}</h1><p>{len(rows)} runs: {counts['ok']} ok, "
            f"<span class=failed>{counts['failed']} failed</span>, <span class=missing>{counts['missing']} not run</span>"
            "</p><table><tr><th>run</th><th>game</th><th>obs</th><th>steering</th><th>status</th><th>episodes</th></tr>"
            f"{table}</table>{''.join(sections)}</body></html>")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs ({counts}) -> {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

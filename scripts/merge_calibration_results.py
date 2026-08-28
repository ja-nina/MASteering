"""Merge per-trait judge calibration results into a single summary.

Run this after all run_judge_calibration.slurm array jobs have finished:

    python scripts/merge_calibration_results.py \\
        --results-root /scratch/inf0/user/nzukowsk/MASteer/judge_calibration \\
        --output-dir   results/judge_calibration_merged
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")

    def _ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (sx * sy) if sx > 1e-12 and sy > 1e-12 else float("nan")


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    return num / (sx * sy) if sx > 1e-12 and sy > 1e-12 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True,
                        help="Directory containing per-trait subdirs with results.jsonl")
    parser.add_argument("--output-dir", default="results/judge_calibration_merged")
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    for jsonl_path in sorted(root.rglob("results.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

    print(f"Total records: {len(all_records)}", flush=True)

    # Write merged JSONL
    merged_path = out / "results_all.jsonl"
    with open(merged_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Per-trait correlation
    trait_records: Dict[str, List[dict]] = {}
    for rec in all_records:
        trait_records.setdefault(rec["trait"], []).append(rec)

    summary = []
    for trait, recs in sorted(trait_records.items()):
        pairs = [
            (r["probe_score"], r["judge_score"])
            for r in recs
            if r.get("probe_score") is not None and r.get("judge_score") is not None
        ]
        if not pairs:
            continue
        probes, judges = zip(*pairs)
        probes, judges = list(probes), list(judges)
        n = len(pairs)
        spear = _spearman(probes, judges)
        pear  = _pearson(probes, judges)
        entry = {
            "trait":      trait,
            "n":          n,
            "spearman":   round(spear, 4),
            "pearson":    round(pear, 4),
            "mean_probe": round(sum(probes) / n, 4),
            "mean_judge": round(sum(judges) / n, 4),
        }
        summary.append(entry)

    summary.sort(key=lambda x: -abs(x["spearman"]) if not math.isnan(x["spearman"]) else 0)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = ["trait", "n", "spearman", "pearson", "mean_probe", "mean_judge"]
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    # Print table
    print(f"\n{'trait':<38} {'n':>5} {'spearman':>10} {'pearson':>10}")
    print("-" * 65)
    for e in summary:
        sp = f"{e['spearman']:+.3f}" if not math.isnan(e["spearman"]) else "  nan"
        pe = f"{e['pearson']:+.3f}"  if not math.isnan(e["pearson"])  else "  nan"
        print(f"{e['trait']:<38} {e['n']:>5} {sp:>10} {pe:>10}")

    print(f"\nMerged → {merged_path}")
    print(f"Summary → {out / 'summary.json'}")


if __name__ == "__main__":
    main()

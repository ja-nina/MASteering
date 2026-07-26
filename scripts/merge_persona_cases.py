"""Merge per-job result files into case_results.jsonl after all array tasks finish.

Usage:
  python scripts/merge_persona_cases.py [--check]

  --check  Print coverage report only (how many jobs are complete) without merging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR     = Path("logs/persona_impact")
MERGED_PATH = OUT_DIR / "case_results.jsonl"

N_CONDITIONS    = 42
N_CHUNKS        = 9
REPS_PER_CASE   = 20
CASES_PER_CHUNK = 6   # last chunk may have fewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Report coverage without merging")
    args = parser.parse_args()

    job_files = sorted(OUT_DIR.glob("results_cond??_chunk??.jsonl"))
    expected  = N_CONDITIONS * N_CHUNKS

    print(f"Found {len(job_files)} / {expected} job files")

    missing = []
    for ci in range(N_CONDITIONS):
        for ch in range(N_CHUNKS):
            p = OUT_DIR / f"results_cond{ci:02d}_chunk{ch:02d}.jsonl"
            if not p.exists():
                missing.append((ci, ch))

    if missing:
        print(f"  Missing ({len(missing)} files):")
        for ci, ch in missing[:20]:
            print(f"    cond{ci:02d}_chunk{ch:02d}  (array task {ci * N_CHUNKS + ch})")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    if args.check:
        return

    if missing:
        print(f"\nWARNING: {len(missing)} job files missing — merging anyway. "
              "Re-run missing tasks before analysing.")

    total = 0
    with MERGED_PATH.open("w", encoding="utf-8") as out:
        for p in job_files:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1

    print(f"\nMerged {total} records -> {MERGED_PATH}")


if __name__ == "__main__":
    main()

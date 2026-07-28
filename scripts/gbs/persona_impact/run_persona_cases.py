"""Run persona-impact counterfactual experiment.

For each case in cases/gbs/persona_impact_cases.json, inject 42 persona conditions
(plain / plain+ToM / 20 behavioral archetypes / 20 archetypes+ToM) into the
system prompt and run Qwen3-14B 20 times, recording the action distribution.

Designed for a SLURM job array with --condition-idx and --chunk so each job
handles at most CASES_PER_CHUNK * REPS_PER_CASE calls (~120).

Results: logs/persona_impact/case_results.jsonl  (one line per inference, append-safe)

Usage
-----
# Single job (testing):
python scripts/run_persona_cases.py --condition-idx 0 --chunk 0 --reps 2

# Full run via SLURM array (see scripts/run_persona_cases.slurm):
sbatch scripts/run_persona_cases.slurm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[3]))

from dotenv import load_dotenv
load_dotenv()

CASES_PATH      = Path("cases/gbs/persona_impact_cases.json")
PERSONAS_PATH   = Path("config/shared/behavioral_personas.yaml")
OUT_DIR         = Path("logs/persona_impact")
MERGED_PATH     = OUT_DIR / "case_results.jsonl"

REPS_PER_CASE   = 20
CASES_PER_CHUNK = 6   # 6 × 20 = 120 calls per job ≤ 130

MODEL_ID = "Qwen/Qwen3-14B"
MODEL_CFG = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    "max_new_tokens": 64,
}

TOM_INSTRUCTION = (
    "Only as a secondary approach, carefully think through step-by-step what others "
    "might guess and how the contributions of others contribute to the sum of the "
    "group guesses for the mystery number. Consider what roles other agents might be "
    "playing (e.g., guessing higher or lower) and adapt your own adjustment to "
    "complement the group."
)

ACTION_RE = re.compile(r"FINAL GUESS[:\s]*\[?(\d+)\]?", re.IGNORECASE)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_action(text: str) -> int | None:
    m = ACTION_RE.search(text)
    if m:
        v = int(m.group(1))
        return v if 0 <= v <= 50 else None
    nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", text) if 0 <= int(x) <= 50]
    return nums[-1] if nums else None


def all_conditions(personas: dict[str, str]) -> list[dict]:
    """Return ordered list of all 42 conditions."""
    conds = [
        {"name": "plain",     "prefix": "", "tom": False},
        {"name": "plain_tom", "prefix": "", "tom": True},
    ]
    for pname, ptext in personas.items():
        conds.append({"name": pname,           "prefix": ptext.strip(), "tom": False})
        conds.append({"name": f"{pname}_tom",  "prefix": ptext.strip(), "tom": True})
    return conds


def make_system_prompt(base: str, prefix: str, tom: bool) -> str:
    parts = ([prefix] if prefix else []) + [base] + ([TOM_INSTRUCTION] if tom else [])
    return "\n\n".join(parts)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-idx", type=int, required=True,
                        help="Index into the ordered condition list (0-41)")
    parser.add_argument("--chunk", type=int, required=True,
                        help="Which case-chunk to process (0-based)")
    parser.add_argument("--cases-per-chunk", type=int, default=CASES_PER_CHUNK)
    parser.add_argument("--reps", type=int, default=REPS_PER_CASE)
    args = parser.parse_args()

    with open(PERSONAS_PATH, encoding="utf-8") as f:
        personas = yaml.safe_load(f)["personas"]

    conds = all_conditions(personas)
    if not (0 <= args.condition_idx < len(conds)):
        sys.exit(f"--condition-idx must be in [0, {len(conds)-1}]")
    cond = conds[args.condition_idx]

    all_cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    start = args.chunk * args.cases_per_chunk
    end   = min(start + args.cases_per_chunk, len(all_cases))
    cases = all_cases[start:end]
    if not cases:
        print(f"Chunk {args.chunk} is empty (only {len(all_cases)} cases total). Exiting.")
        return

    # Per-job output file — no concurrent writes
    job_out = OUT_DIR / f"results_cond{args.condition_idx:02d}_chunk{args.chunk:02d}.jsonl"

    n_calls = len(cases) * args.reps
    print(f"Condition [{args.condition_idx}] {cond['name']}  |  "
          f"chunk {args.chunk}: cases {start}-{end-1}  |  {n_calls} calls")

    # ── W&B ──
    try:
        import wandb
        run = wandb.init(
            project="ma-steering-persona-impact",
            name=f"persona_cases_{cond['name']}_chunk{args.chunk}",
            tags=["persona_cases", cond["name"],
                  "tom" if cond["tom"] else "no_tom",
                  f"chunk{args.chunk}"],
            config={
                "condition":      cond["name"],
                "condition_idx":  args.condition_idx,
                "chunk":          args.chunk,
                "cases":          [c["case_id"] for c in cases],
                "reps":           args.reps,
                "model":          MODEL_ID,
                "tom":            cond["tom"],
            },
        )
        wb_table = wandb.Table(columns=[
            "case_id", "category", "round", "target",
            "condition", "rep", "action", "original_action", "delta",
        ])
    except ImportError:
        print("wandb not installed — skipping.")
        run = None
        wb_table = None

    # ── resume: if this job's file already exists, skip completed reps ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, int]] = set()
    if job_out.exists():
        for line in job_out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["case_id"], r["condition"], r["rep"]))
    skipped = sum(1 for c in cases for rep in range(args.reps)
                  if (c["case_id"], cond["name"], rep) in done)
    print(f"Skipping {skipped} already-done inferences")

    # ── load model ──
    print(f"Loading {MODEL_ID} ...")
    from testbed.policy.transformers_policy import TransformersPolicy
    policy = TransformersPolicy(model_id=MODEL_ID, enable_thinking=False, **MODEL_CFG)

    out_fh = job_out.open("a", encoding="utf-8")
    done_now = 0

    for case in cases:
        sys_prompt = make_system_prompt(case["system_prompt"], cond["prefix"], cond["tom"])

        for rep in range(args.reps):
            if (case["case_id"], cond["name"], rep) in done:
                continue

            result     = policy.act(sys_prompt, case["user_prompt"],
                                    agent_id="player_0", steering=None)
            completion = result[0] if isinstance(result, tuple) else result
            action     = parse_action(completion)
            delta      = (action - case["original_action"]) if action is not None else None

            rec = {
                "case_id":        case["case_id"],
                "category":       case["category"],
                "round":          case["round"],
                "target":         case["target"],
                "condition":      cond["name"],
                "condition_idx":  args.condition_idx,
                "rep":            rep,
                "action":         action,
                "original_action": case["original_action"],
                "delta":          delta,
                "completion":     completion,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()

            if wb_table is not None and action is not None:
                wb_table.add_data(
                    case["case_id"], case["category"], case["round"],
                    case["target"], cond["name"], rep, action,
                    case["original_action"], delta,
                )

            done_now += 1
            if done_now % 20 == 0:
                print(f"  {done_now}/{n_calls - skipped}  "
                      f"case={case['case_id']}  action={action}")

    out_fh.close()

    if run is not None:
        run.log({"results": wb_table})
        # per-condition summary stats
        if wb_table and wb_table.data:
            actions = [row[6] for row in wb_table.data if row[6] is not None]
            deltas  = [row[8] for row in wb_table.data if row[8] is not None]
            import statistics as st
            run.summary.update({
                "n_inferences":  len(actions),
                "mean_action":   st.mean(actions) if actions else None,
                "std_action":    st.stdev(actions) if len(actions) > 1 else 0,
                "mean_delta":    st.mean(deltas) if deltas else None,
                "parse_rate":    len(actions) / max(done_now, 1),
            })
        run.finish()

    print(f"\nDone. Results written to {job_out}")


if __name__ == "__main__":
    main()

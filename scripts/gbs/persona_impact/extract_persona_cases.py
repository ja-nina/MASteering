"""Extract 50 interesting game states from plain_10p_14b logs for persona impact study.

Each "case" is a single agent decision point (player_0 at a specific round) with
its full (system_prompt, user_prompt) captured from the plain condition — so there
is no persona prefix in the system_prompt, giving a clean baseline to inject into.

Categories (50 total):
  oscillating      (15) — feedback alternates HIGH/LOW for 3+ rounds
  stuck            (10) — same feedback direction 4+ rounds in a row
  near_convergence (10) — group was within 10% of target last round
  late_game        (10) — round >= 18, still not converged
  first_round       (5) — round 1, diverse targets

Output: cases/gbs/persona_impact_cases.json
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path

LOG_DIR  = Path("logs/picking_sweep/gbs_exact_replication_plain_10p_14b")
OUT_PATH = Path("cases/gbs/persona_impact_cases.json")
SEED     = 42
N_PER_CATEGORY = {"oscillating": 15, "stuck": 10, "near_convergence": 10,
                  "late_game": 10, "first_round": 5}


def parse_feedback_history(user_prompt: str) -> list[str]:
    return re.findall(r"Result: (too HIGH|too LOW)", user_prompt)


def parse_round_number(user_prompt: str) -> int:
    m = re.search(r"Round (\d+) of", user_prompt)
    return int(m.group(1)) if m else 0


def classify(history: list[str], round_num: int, prev_error: int | None,
             prev_target: int | None) -> list[str]:
    cats = []
    if round_num == 1:
        cats.append("first_round")
    if len(history) >= 3:
        last3 = history[-3:]
        if last3[0] != last3[1] and last3[1] != last3[2]:
            cats.append("oscillating")
    if len(history) >= 4 and len(set(history[-4:])) == 1:
        cats.append("stuck")
    if round_num >= 18:
        cats.append("late_game")
    if prev_error is not None and prev_target is not None and prev_target > 0:
        if abs(prev_error) / prev_target <= 0.10:
            cats.append("near_convergence")
    return cats


def main():
    rng = random.Random(SEED)
    candidates: dict[str, list[dict]] = {c: [] for c in N_PER_CATEGORY}

    for ep_jsonl in sorted(LOG_DIR.glob("episode_*.jsonl")):
        ep_num = int(re.search(r"episode_(\d+)", ep_jsonl.name).group(1))
        turns_raw = [json.loads(l) for l in ep_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

        # Keep only player_0 turns, sorted by turn index
        p0_turns = sorted([t for t in turns_raw if t["agent_id"] == "player_0"],
                          key=lambda t: t["turn"])

        prev_error  = None
        prev_target = None

        for idx, rec in enumerate(p0_turns):
            history  = parse_feedback_history(rec["user_prompt"])
            round_n  = parse_round_number(rec["user_prompt"])
            cats     = classify(history, round_n, prev_error, prev_target)

            # Infer target from info: error = group_sum - target
            info = rec.get("info", {})
            if "error" in info and "group_sum" in info:
                target = info["group_sum"] - info["error"]
                prev_error  = info["error"]
                prev_target = target
            else:
                target = None

            case = {
                "case_id":        f"ep{ep_num}_r{round_n}",
                "episode":        ep_num,
                "round":          round_n,
                "system_prompt":  rec["system_prompt"],
                "user_prompt":    rec["user_prompt"],
                "original_action": rec["parsed_action"],
                "feedback_history": history,
                "target":         target,
                "info_after":     info,
            }

            for cat in cats:
                if cat in candidates:
                    candidates[cat].append(case)

    # Sample N per category without cross-category duplicates
    selected: list[dict] = []
    used_ids: set[str]   = set()

    for cat, n in N_PER_CATEGORY.items():
        pool = [c for c in candidates[cat] if c["case_id"] not in used_ids]
        rng.shuffle(pool)
        picked = pool[:n]
        for c in picked:
            c["category"] = cat
            used_ids.add(c["case_id"])
        selected.extend(picked)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Extracted {len(selected)} cases -> {OUT_PATH}")
    for cat in N_PER_CATEGORY:
        n = sum(1 for c in selected if c["category"] == cat)
        print(f"  {cat:<20} {n:>3} / {N_PER_CATEGORY[cat]}")


if __name__ == "__main__":
    main()

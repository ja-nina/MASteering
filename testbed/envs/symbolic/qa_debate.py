"""Accuracy debate (Du et al. 2023, 'Improving factuality and reasoning through multiagent debate'): N agents answer
the same question; from round 2 on each sees the others' previous answers and reasoning and may revise. The group's
answer is the most common final-round answer. Rewards are correctness, so steering is scored on a task with a right
answer instead of a payoff.

Questions: testbed/data/<dataset>.jsonl (scripts/make_debate_data.py), a fixed shuffled sample. Episode i plays
question i at every player count and in every run (the runner passes the episode index: uses_episode), so all
conditions and all N answer the same questions.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult

DATA = Path(__file__).resolve().parents[2] / "data"
LETTERS = "ABCDEFGHIJ"
MULTIPLE_CHOICE = ("mmlu_pro", "supergpqa")  # letter answers; gsm8k: a number
DATASETS = ("gsm8k",) + MULTIPLE_CHOICE


@lru_cache(maxsize=None)
def questions(dataset: str) -> tuple:
    with open(DATA / f"{dataset}.jsonl", encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)


def normalize(answer: Optional[str], dataset: str) -> Optional[str]:
    """Canonical form of an answer (None if there is none): a letter for multiple choice, a number for GSM8K."""
    if answer is None:
        return None
    a = str(answer).strip()
    if dataset in MULTIPLE_CHOICE:
        return a.upper() if re.fullmatch(r"[A-Ja-j]", a) else None
    a = a.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        x = float(a)
    except ValueError:
        return None
    return str(int(x)) if x == int(x) else f"{x:g}"


class QADebateAdapter(SymbolicAdapter):
    uses_episode = True  # the runner passes episode=<index>: question i in episode i, at every N

    def __init__(self, num_players: int = 3, dataset: str = "gsm8k", num_rounds: int = 3, seed: int = 0,
                 episode: Optional[int] = None, show_chars: int = 1500) -> None:
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}: {' | '.join(DATASETS)}")
        pool = questions(dataset)
        self.dataset = dataset
        self.index = (episode if episode is not None else seed) % len(pool)
        self.item = pool[self.index]
        self.correct = normalize(self.item["answer"], dataset)
        self.show_chars = show_chars
        self.final: Dict[str, Optional[str]] = {}

    def _observation(self, agent_id: str) -> RawObs:
        return {
            "agent_id": agent_id,
            "round_index": self.context.round_index,
            "num_rounds": self.num_rounds,
            "num_players": self.num_players,
            "dataset": self.dataset,
            "question": self.item["question"],
            "options": list(self.item.get("options") or []),
            "history": list(self.context.history),
            "show_chars": self.show_chars,
        }

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        answers, texts = {}, {}
        for pid in self._ids:
            act = actions.get(pid)
            if isinstance(act, dict):  # the parser's {answer, text}; anything else is the runner's fallback
                answers[pid], texts[pid] = normalize(act.get("answer"), self.dataset), str(act.get("text") or "")
            else:
                answers[pid], texts[pid] = None, ""
        correct = {pid: answers[pid] is not None and answers[pid] == self.correct for pid in self._ids}
        counts = Counter(a for a in answers.values() if a is not None)
        top = max(counts.values()) if counts else 0
        tied = sorted(a for a, k in counts.items() if k == top)
        # a tie counts as the share of tied answers that are right (a random tie-break's expected accuracy)
        majority_correct = (sum(a == self.correct for a in tied) / len(tied)) if tied else 0.0

        self.context.round_index += 1
        done = self.context.round_index >= self.num_rounds
        rewards = {pid: (1.0 if correct[pid] else 0.0) if done else 0.0 for pid in self._ids}
        self.context.last_rewards = rewards
        info = {"round": self.context.round_index, "question_id": self.item["id"], "correct_answer": self.correct,
                "answers": answers, "correct": correct, "n_correct": sum(correct.values()),
                "majority": tied[0] if len(tied) == 1 else None, "majority_tied": tied if len(tied) > 1 else [],
                "majority_correct": majority_correct}
        if self.item.get("category"):
            info["category"] = self.item["category"]
        self.context.history.append({**info, "texts": texts})
        if done:
            self.final = answers
        return StepResult(rewards=rewards, done=done, info=info)

    def game_scores(self) -> Optional[Dict[str, float]]:
        """1 if a player's final answer is right, else 0 (welfare = number of right final answers)."""
        if not self.final:
            return None
        return {pid: float(self.final[pid] is not None and self.final[pid] == self.correct) for pid in self._ids}

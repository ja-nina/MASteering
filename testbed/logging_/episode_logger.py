"""JSONL trace logger: one line per agent per turn, plus an episode summary."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class EpisodeLogger:
    def __init__(self, run_dir: str, run_id: str, episode: int) -> None:
        self.dir = os.path.join(run_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = run_id
        self.episode = episode
        self.path = os.path.join(self.dir, f"episode_{episode}.jsonl")
        self._fh = open(self.path, "w", encoding="utf-8")

    def log_step(self, *, game: str, turn: int, agent_id: str, system_prompt: str,
                 user_prompt: str, completion: str, parsed_action: Any,
                 parse_retries: int, reward: float,
                 steering_spec_id: Optional[str]) -> None:
        rec = {
            "run_id": self.run_id, "episode": self.episode, "game": game,
            "turn": turn, "agent_id": agent_id, "system_prompt": system_prompt,
            "user_prompt": user_prompt, "completion": completion,
            "parsed_action": parsed_action, "parse_retries": parse_retries,
            "reward": reward, "steering_spec_id": steering_spec_id,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if not self._fh.closed:
            self._fh.close()
        if summary is not None:
            spath = os.path.join(self.dir, f"episode_{self.episode}.summary.json")
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

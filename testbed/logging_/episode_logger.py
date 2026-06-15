"""JSONL trace logger: one line per agent per turn, plus an episode summary.

Pass a wandb run object to also stream metrics and upload the trace as an artifact.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


_SEP = "=" * 72


class EpisodeLogger:
    def __init__(self, run_dir: str, run_id: str, episode: int,
                 wandb_run=None) -> None:
        self.dir = os.path.join(run_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = run_id
        self.episode = episode
        self.path = os.path.join(self.dir, f"episode_{episode}.jsonl")
        self.trace_path = os.path.join(self.dir, f"episode_{episode}.trace.txt")
        self._fh = open(self.path, "w", encoding="utf-8")
        self._trace = open(self.trace_path, "w", encoding="utf-8")
        self._wandb = wandb_run
        self._global_step = 0

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

        # human-readable trace
        self._trace.write(
            f"{_SEP}\n"
            f"Turn {turn:>3} | {agent_id} | game={game} | "
            f"steering={steering_spec_id}\n"
            f"{_SEP}\n"
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{user_prompt}\n\n"
            f"[COMPLETION]\n{completion}\n\n"
            f"[ACTION={parsed_action}  REWARD={reward:.3f}"
            f"  RETRIES={parse_retries}]\n\n"
        )
        self._trace.flush()

        if self._wandb is not None:
            self._wandb.log({
                f"{agent_id}/reward": reward,
                f"{agent_id}/parse_retries": parse_retries,
                "turn": turn,
                "episode": self.episode,
                "steering": steering_spec_id,
            }, step=self._global_step)
        self._global_step += 1

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if not self._fh.closed:
            self._fh.close()
        if not self._trace.closed:
            self._trace.close()
        if summary is not None:
            spath = os.path.join(self.dir, f"episode_{self.episode}.summary.json")
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            if self._wandb is not None:
                # episode-level summary metrics
                for agent_id, r in summary.get("final_rewards", {}).items():
                    self._wandb.summary[f"final_reward/{agent_id}"] = r
                # upload full trace as an artifact
                artifact = self._wandb.Artifact(
                    f"episode-{self.episode}-trace", type="trace")
                artifact.add_file(self.path)
                self._wandb.log_artifact(artifact)

"""JSONL trace logger: one line per agent per turn, a readable trace, and an episode summary.

Each JSONL record carries what the agent was given and what it said:
  system_prompt   the system prompt, persona prompt included when the agent has one
  user_prompt     the user turn of the final attempt (with parser feedback after a retry)
  completion      the final attempt's output
  attempts        every attempt in order: its user prompt, completion and parse error (if any)
  model_input     the exact chat-templated text of the final attempt (image placeholders included)
and `steering` (the agent's steering config, null when unsteered),
`monitor` ({layer: {trait: projection}}, empty when monitoring is off) and `images` (paths of
the turn's image observation, relative to the run dir, when save_images is on; else a count).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_SEP = "=" * 72


class EpisodeLogger:
    def __init__(self, run_dir: str, run_id: str, episode: int, save_images: bool = False) -> None:
        self.dir = os.path.join(run_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = run_id
        self.episode = episode
        self.path = os.path.join(self.dir, f"episode_{episode}.jsonl")
        self.trace_path = os.path.join(self.dir, f"episode_{episode}.trace.txt")
        self._fh = open(self.path, "w", encoding="utf-8")
        self._trace = open(self.trace_path, "w", encoding="utf-8")
        self.save_images = save_images

    def log_step(self, *, game: str, turn: int, agent_id: str, system_prompt: str,
                 user_prompt: str, completion: str, parsed_action: Any,
                 parse_retries: int, reward: float, steering: Optional[Dict[str, Any]] = None,
                 info: Optional[Dict[str, Any]] = None, parse_ok: bool = True,
                 finish_reason: str = "stop", n_tokens: int = 0,
                 monitor: Optional[Dict[str, Dict[str, float]]] = None, images=(),
                 attempts: Optional[list] = None, model_input: str = "") -> None:
        image_log = self._save_images(turn, agent_id, images) if self.save_images else len(images)
        rec = {
            "run_id": self.run_id, "episode": self.episode, "game": game,
            "turn": turn, "agent_id": agent_id, "system_prompt": system_prompt,
            "user_prompt": user_prompt, "completion": completion,
            "parsed_action": parsed_action, "parse_retries": parse_retries, "parse_ok": parse_ok,
            "reward": reward, "steering": steering, "finish_reason": finish_reason,
            "n_tokens": n_tokens, "info": info or {}, "monitor": monitor or {}, "images": image_log,
            "attempts": attempts or [], "model_input": model_input,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

        label = steering["label"] if steering else "none"
        shown = f"[IMAGES] {image_log}\n\n" if images else ""
        rejected = "".join(f"[ATTEMPT {a['attempt'] + 1} REJECTED: {a['parse_error']}]\n[USER]\n{a['user_prompt']}\n\n"
                           f"[COMPLETION]\n{a['completion']}\n\n"
                           for a in (attempts or [])[:-1])  # every attempt but the final one was rejected
        self._trace.write(
            f"{_SEP}\nTurn {turn:>3} | {agent_id} | game={game} | steering={label}\n{_SEP}\n"
            f"[SYSTEM]\n{system_prompt}\n\n{rejected}[USER]\n{user_prompt}\n\n{shown}[COMPLETION]\n{completion}\n\n"
            f"[ACTION={parsed_action}  REWARD={reward:.3f}  RETRIES={parse_retries}"
            f"{'' if parse_ok else '  FALLBACK'}{'  TRUNCATED' if finish_reason == 'length' else ''}]\n\n")
        if info:
            self._trace.write(f"[INFO] {json.dumps(info, default=str)}\n\n")
        if monitor:
            layer, scores = max(monitor.items(), key=lambda kv: int(kv[0]))
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
            self._trace.write(f"[MONITOR L{layer} top-5] {', '.join(f'{t}={v:.2f}' for t, v in top)}\n\n")
        self._trace.flush()

    def _save_images(self, turn: int, agent_id: str, images) -> list:
        paths = []
        for i, image in enumerate(images):
            rel = f"episode_{self.episode}_images/t{turn:04d}_{agent_id}_{i}.png"
            os.makedirs(os.path.join(self.dir, os.path.dirname(rel)), exist_ok=True)
            image.save(os.path.join(self.dir, rel))
            paths.append(rel)
        return paths

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Close the files; the summary file (written last) marks the episode as complete."""
        if not self._fh.closed:
            self._fh.close()
        if not self._trace.closed:
            self._trace.close()
        if summary is not None:
            spath = os.path.join(self.dir, f"episode_{self.episode}.summary.json")
            tmp = spath + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, spath)

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

The summary (written last) also gets `agent_stats`: per agent its turns, parse fallbacks, answer chars and
mean monitor projection per layer and trait, so analyses never need to read the (large) JSONL.
"""
from __future__ import annotations

import errno
import json
import os
import sys
import time
from typing import Any, Dict, Optional

_SEP = "=" * 72
# network filesystems (cluster scratch) drop out for seconds to minutes: wait those out instead of failing
_TRANSIENT = {getattr(errno, name, code) for name, code in  # Linux numbers where the platform lacks the name
              (("ECOMM", 70), ("EIO", 5), ("ESTALE", 116), ("ETIMEDOUT", 110), ("EAGAIN", 11), ("EHOSTDOWN", 112))}
_RETRY_S = float(os.environ.get("MA_FS_RETRY_S", 600))


# a log root that just failed is skipped (episodes go to the fallback) for this long
_DOWN_S = float(os.environ.get("MA_FS_DOWN_S", 900))
_DOWN_UNTIL: Dict[str, float] = {}


def retry_io(fn, *args, patience: Optional[float] = None, **kwargs):
    """fn(*args, **kwargs), retried with backoff on transient I/O errors for up to `patience` seconds
    (default MA_FS_RETRY_S)."""
    patience = _RETRY_S if patience is None else patience
    waited, delay = 0.0, 2.0
    while True:
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if e.errno not in _TRANSIENT or waited >= patience:
                raise
            print(f"[fs] {e} - retrying in {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, 60.0)


class EpisodeLogger:
    """Writes to `run_dir`/<run_id>; if that is unusable (after a minute of retries) and `fallback_dir` is
    given, to `fallback_dir`/<run_id> instead, with a copy of `run_info` as run.json there. A failed
    run_dir is then skipped for MA_FS_DOWN_S seconds, so later episodes go straight to the fallback."""

    def __init__(self, run_dir: str, run_id: str, episode: int, save_images: bool = False,
                 fallback_dir: Optional[str] = None, run_info: Optional[Dict[str, Any]] = None) -> None:
        self.run_id = run_id
        self.episode = episode
        self.save_images = save_images
        self._stats: Dict[str, Dict[str, Any]] = {}
        roots = [str(run_dir)] + ([str(fallback_dir)] if fallback_dir else [])
        for i, root in enumerate(roots):
            last = i == len(roots) - 1
            if not last and _DOWN_UNTIL.get(root, 0.0) > time.time():
                continue
            try:
                self._open(root, patience=None if last else 60.0)
            except OSError as e:
                if last:
                    raise
                _DOWN_UNTIL[root] = time.time() + _DOWN_S
                print(f"[fs] {root} unusable ({e}): episodes go to {roots[i + 1]} for the next "
                      f"{_DOWN_S / 60:.0f} min", file=sys.stderr, flush=True)
                continue
            if i > 0 and run_info is not None:
                retry_io(self._write_json, os.path.join(self.dir, "run.json"), run_info)
            break

    def _open(self, root: str, patience: Optional[float]) -> None:
        self.dir = os.path.join(root, self.run_id)
        retry_io(os.makedirs, self.dir, exist_ok=True, patience=patience)
        self.path = os.path.join(self.dir, f"episode_{self.episode}.jsonl")
        self.trace_path = os.path.join(self.dir, f"episode_{self.episode}.trace.txt")
        self._fh = retry_io(open, self.path, "w", encoding="utf-8", patience=patience)
        self._trace = retry_io(open, self.trace_path, "w", encoding="utf-8", patience=patience)

    def log_step(self, *, game: str, turn: int, agent_id: str, system_prompt: str,
                 user_prompt: str, completion: str, parsed_action: Any,
                 parse_retries: int, reward: float, steering: Optional[Dict[str, Any]] = None,
                 info: Optional[Dict[str, Any]] = None, parse_ok: bool = True,
                 finish_reason: str = "stop", n_tokens: int = 0,
                 monitor: Optional[Dict[str, Dict[str, float]]] = None, images=(),
                 attempts: Optional[list] = None, model_input: str = "") -> None:
        st = self._stats.setdefault(agent_id, {"turns": 0, "fallbacks": 0, "chars": 0, "monitor": {}})
        st["turns"] += 1
        st["fallbacks"] += not parse_ok
        st["chars"] += len(completion or "")
        for layer, scores in (monitor or {}).items():  # running sums and counts; means in agent_stats()
            for trait, v in scores.items():
                acc = st["monitor"].setdefault(str(layer), {}).setdefault(trait, [0.0, 0])
                acc[0] += v
                acc[1] += 1
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
        retry_io(self._fh.flush)

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
        retry_io(self._trace.flush)

    def _save_images(self, turn: int, agent_id: str, images) -> list:
        paths = []
        for i, image in enumerate(images):
            rel = f"episode_{self.episode}_images/t{turn:04d}_{agent_id}_{i}.png"
            retry_io(os.makedirs, os.path.join(self.dir, os.path.dirname(rel)), exist_ok=True)
            retry_io(image.save, os.path.join(self.dir, rel))
            paths.append(rel)
        return paths

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Close the files; the summary file (written last) marks the episode as complete."""
        if not self._fh.closed:
            self._fh.close()
        if not self._trace.closed:
            self._trace.close()
        if summary is not None:
            summary = {**summary, "agent_stats": self.agent_stats()}
            spath = os.path.join(self.dir, f"episode_{self.episode}.summary.json")
            tmp = spath + f".{os.getpid()}.tmp"
            retry_io(self._write_json, tmp, summary)
            retry_io(os.replace, tmp, spath)

    def agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """{agent: {turns, fallbacks, chars, monitor: {layer: {trait: mean projection over its turns}}}}."""
        return {aid: {**{k: st[k] for k in ("turns", "fallbacks", "chars")},
                      "monitor": {layer: {t: total / n for t, (total, n) in traits.items()}
                                  for layer, traits in st["monitor"].items()}}
                for aid, st in self._stats.items()}

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

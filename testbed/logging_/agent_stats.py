"""Per-agent statistics of one episode, as stored in its summary (`agent_stats`), so analyses never need the
(large) JSONL. The episode logger feeds every turn in; scripts/summarize_runs.py feeds the JSONL of episodes
logged before (same numbers either way).

Per agent:
  turns, fallbacks, chars   turns, turns whose answer stayed unparseable, characters answered
  monitor                   {layer: {trait: mean projection over its turns}}
  phase_layer, phases       the monitor at the middle layer, averaged over the first, middle and last third of
                            the agent's own monitored turns: [{trait: mean} or None] * 3 (None: too few turns)
  actions                   what it chose, over its parsed turns (default moves after a failed parse excluded):
                            {"n": count, "numeric": mean of numeric choices (bids, guesses, amounts) or None,
                             "labels": {action label: count}} (labels: see action_label)
"""
from __future__ import annotations

import json
import re
import statistics
from typing import Any, Dict, Optional

_BRACKET = re.compile(r"\[([^\[\]]{1,40})\]")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
MAX_LABELS = 30


def action_label(value: Any) -> tuple:
    """(label, number or None) of a parsed action, comparable across agents of a game: APPROVE / REJECT,
    play / discard / hint:color, 'bid #' for TextArena's '[Bid 300]' (the number goes to `number`), player_#
    for a vote target, 'number' for a plain number, 'move' for a board cell, 'text' for free text."""
    if isinstance(value, bool):
        return str(value).lower(), None
    if isinstance(value, (int, float)):
        return "number", float(value)
    if isinstance(value, dict):
        if "type" in value:
            return value["type"] + (f":{value['hint_type']}" if value.get("hint_type") else ""), None
        return "+".join(sorted(value))[:40], None
    if isinstance(value, (list, tuple)):
        return "move", None
    if isinstance(value, str):
        brackets = _BRACKET.findall(value)
        if brackets:  # TextArena: the last bracketed command is the action
            cmd = brackets[-1].strip()
            num = _NUMBER.search(cmd)
            return _NUMBER.sub("#", cmd.lower())[:40], float(num.group(0)) if num else None
        if len(value) <= 40 and "\n" not in value:
            return _NUMBER.sub("#", value.strip().lower()), None
        return "text", None
    return type(value).__name__, None


class AgentStats:
    def __init__(self) -> None:
        self._a: Dict[str, Dict[str, Any]] = {}

    def add(self, agent_id: str, *, parse_ok: bool = True, completion: str = "",
            monitor: Optional[Dict[str, Dict[str, float]]] = None, action: Any = None) -> None:
        a = self._a.setdefault(agent_id, {"turns": 0, "fallbacks": 0, "chars": 0, "monitor": [],
                                          "labels": {}, "numbers": [], "n": 0})
        a["turns"] += 1
        a["fallbacks"] += not parse_ok
        a["chars"] += len(completion or "")
        if monitor:
            a["monitor"].append({str(layer): scores for layer, scores in monitor.items()})
        if parse_ok and action is not None:
            label, number = action_label(action)
            a["n"] += 1
            a["labels"][label] = a["labels"].get(label, 0) + 1
            if number is not None:
                a["numbers"].append(number)

    def add_record(self, rec: Dict[str, Any]) -> None:
        """One JSONL record (episode_logger's format)."""
        self.add(rec["agent_id"], parse_ok=rec.get("parse_ok", True), completion=rec.get("completion") or "",
                 monitor=rec.get("monitor"), action=rec.get("parsed_action"))

    def to_json(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for aid, a in self._a.items():
            layers = sorted({layer for turn in a["monitor"] for layer in turn}, key=int)
            mid = layers[len(layers) // 2] if layers else None
            labels = dict(sorted(a["labels"].items(), key=lambda kv: -kv[1])[:MAX_LABELS])
            out[aid] = {
                "turns": a["turns"], "fallbacks": a["fallbacks"], "chars": a["chars"],
                "monitor": {layer: _means([t[layer] for t in a["monitor"] if layer in t]) for layer in layers},
                "phase_layer": mid, "phases": _phases([t[mid] for t in a["monitor"] if mid in t]) if mid else [None] * 3,
                "actions": {"n": a["n"], "numeric": statistics.mean(a["numbers"]) if a["numbers"] else None,
                            "labels": labels},
            }
        return out


def _means(turns) -> Dict[str, float]:
    sums: Dict[str, list] = {}
    for scores in turns:
        for trait, v in scores.items():
            sums.setdefault(trait, []).append(v)
    return {t: round(statistics.mean(v), 5) for t, v in sums.items()}


def _phases(turns) -> list:
    """Means over the first, middle and last third of the turns (turn i of k goes to third 3i // k)."""
    k = len(turns)
    return [(_means([s for i, s in enumerate(turns) if 3 * i // k == p]) or None) if k else None for p in range(3)]


def stats_from_jsonl(path) -> Dict[str, Dict[str, Any]]:
    stats = AgentStats()
    with open(path, encoding="utf-8") as f:
        for line in f:
            stats.add_record(json.loads(line))
    return stats.to_json()

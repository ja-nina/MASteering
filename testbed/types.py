"""Core shared types for the testbed."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Game-specific payloads kept loose on purpose: symbolic games use dicts,
# TextArena uses the observation string, etc.
RawObs = Any
Action = Any


@dataclass
class StepResult:
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedAction:
    value: Action


@dataclass
class ParseError:
    feedback: str


ParseResult = Union[ParsedAction, ParseError]


@dataclass
class SteeringSpec:
    method: str  # "noop" | "prompt_injection" | "activation"
    layer: Optional[str] = None
    vector_path: Optional[str] = None
    coefficient: float = 0.0
    # free-form params for prompt injection (e.g. {"system_suffix": "..."})
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderContext:
    round_index: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_rewards: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

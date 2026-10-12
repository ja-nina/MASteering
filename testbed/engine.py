"""What the game runner needs from a model, independent of vLLM.

    Generator.generate(requests) -> results      one batched call per scheduler step
    Reader.read(results)          -> residuals    the completion's residual stream at the
                                                  monitor layers (rows per layer)

testbed/mia_engine.py implements both with vLLM-Hook; tests use tests/fakes.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import torch

from testbed.steering import Steer


@dataclass
class TurnRequest:
    key: tuple                    # (run_index, episode, turn, agent_id, attempt)
    system: str
    user: str
    seed: int
    steer: Optional[Steer] = None
    max_tokens: Optional[int] = None
    images: List = field(default_factory=list)   # PIL images shown before the user text


@dataclass
class TurnResult:
    key: tuple
    text: str
    finish_reason: str
    prompt_token_ids: List[int] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    prompt: str = ""                              # templated prompt text (needed to re-read image turns)
    images: List = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


class Generator(Protocol):
    def generate(self, requests: List[TurnRequest]) -> List[TurnResult]: ...


class Reader(Protocol):
    layers: List[int]

    def read(self, results: List[TurnResult]) -> List[Dict[int, torch.Tensor]]:
        """Per result: {layer: [n_completion_tokens, hidden]} residual rows (post-block, 0-based layer)."""
        ...


class Monitor:
    """Projects a completion's mean residual onto unit persona vectors at a few layers."""

    def __init__(self, store, layers: List[int], traits: Optional[List[str]] = None) -> None:
        self.layers = list(layers)
        self.traits = list(traits) if traits else store.traits()
        units = []
        for layer in self.layers:
            vs = torch.stack([store.vectors(t)[layer] for t in self.traits])
            units.append(vs / vs.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        self._units = torch.stack(units)        # [n_layers, n_traits, hidden]

    def scores(self, rows: Dict[int, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """{layer: {trait: mean over completion tokens of h . v_hat}}; {} for an empty completion."""
        if any(rows[l].shape[0] == 0 for l in self.layers):
            return {}
        out = {}
        for i, layer in enumerate(self.layers):
            proj = self._units[i] @ rows[layer].float().mean(0)
            out[str(layer)] = {t: round(float(p), 4) for t, p in zip(self.traits, proj)}
        return out

"""GPU-free stand-ins for the vLLM-Hook engines and a PersVecFast vector directory."""
from __future__ import annotations

import random
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from testbed.engine import TurnResult
from testbed.envs.symbolic.werewolf import _NAMES

HIDDEN, LAYERS, TRAITS = 8, 6, ("evil", "sarcastic", "empathy")


def make_vectors(root: Path) -> Path:
    """A PersVecFast-style output dir: {trait}.pt (all layers), {trait}.json, vllm_hook/ files."""
    root.mkdir(parents=True, exist_ok=True)
    for i, trait in enumerate(TRAITS):
        vectors = {l: torch.full((HIDDEN,), float(i + 1)) * (l + 1) for l in range(LAYERS)}
        vectors = {l: v * torch.tensor([1.0 if j == i else 0.1 for j in range(HIDDEN)]) for l, v in vectors.items()}
        torch.save(vectors, root / f"{trait}.pt")
        best = 1 + i % 3  # evil -> 1, sarcastic -> 2, empathy -> 3
        sweep = {str(l): {"0.0": 10.0, "1.25": 20.0, "2.5": 30.0 + (20 if l == best else 0)} for l in (1, 2, 3)}
        (root / f"{trait}.json").write_text(
            __import__("json").dumps({"trait": trait, "layers": [1, 2, 3], "sweep_scores": sweep}))
        (root / "vllm_hook").mkdir(exist_ok=True)
        for l in (1, 2, 3):
            torch.save({"dir": vectors[l]}, root / "vllm_hook" / f"{trait}_L{l}.pt")
    torch.save({}, root / "evil_raw.pt")  # PersVecFast also writes _raw.pt files: not a trait
    traits = root.parent / "traits"  # PersVecFast/data/persona_vectors stand-in
    traits.mkdir(exist_ok=True)
    for trait in TRAITS:
        (traits / f"{trait}.json").write_text(__import__("json").dumps({"trait": trait, "instruction": {
            "positive": f"You clearly exhibit the trait '{trait}'.",
            "negative": f"You do NOT exhibit the trait '{trait}'."}}))
    return root


def reply(request) -> str:
    """A plausible move for any of the symbolic games, chosen by the prompt's instructions."""
    rng = random.Random(request.seed)
    user = request.user
    if "FINAL GUESS" in user or "NUMBER:" in user:
        return f"NUMBER: {rng.randint(0, 60)}"
    if "CHOICE:" in user:
        return f"CHOICE: {rng.randint(0, 100)}"
    if "ACTIONS:" in user:
        return f"ACTIONS: {rng.choice(['FORWARD', 'TURN_LEFT', 'FIRE_ZAP'])}, FORWARD"
    if "MOVE:" in user:
        return f"MOVE: {rng.choice('ABCDEFGHI')}{rng.randint(1, 9)}"
    if "MESSAGE:" in user:
        return "MESSAGE: let us each take 10 tons."
    if "HARVEST:" in user:
        return f"HARVEST: {rng.randint(0, 15)}"
    names = [n for n in _NAMES if re.search(rf"\b{n}\b", user) and n != request.key[3]]
    target = rng.choice(names) if names else "nobody"
    return f"KILL: {target}\nINVESTIGATE: {target}\nSAVE: {target}\nVOTE: {target}\nSTATEMENT: I suspect {target}."


class FakeGenerator:
    def __init__(self, respond=reply) -> None:
        self.respond = respond
        self.batches = []

    def generate(self, requests):
        self.batches.append(list(requests))
        return [TurnResult(r.key, self.respond(r), "stop", prompt_token_ids=[1, 2, 3],
                           token_ids=list(range(len(self.respond(r).split()))),
                           prompt=f"<system>{r.system}</system><user>{r.user}</user>", images=r.images)
                for r in requests]


class FakeReader:
    def __init__(self, layers) -> None:
        self.layers = list(layers)
        self.calls = 0

    def read(self, results):
        self.calls += 1
        return [{l: torch.ones(r.n_tokens, HIDDEN) for l in self.layers} for r in results]


def fake_open_engines(generator=None):
    @contextmanager
    def open_engines(model, engine_cfg, layers):
        yield (generator or FakeGenerator()), (FakeReader(layers) if layers is not None else None)
    return open_engines

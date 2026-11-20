"""Persona steering: vectors (vLLM-Hook's add_vector op) and/or persona prompts.

Every generation request carries its own steering config, so in one batch some
agents are steered and others are not. An agent's config is a `Steer` with up to
two channels, chosen per group by `via`:
    vector  hidden state += coeff * dir at one decoder block (0-based), where `dir`
            is the raw (unnormalised) persona vector, as PersVecFast evaluates it
    prompt  the trait's PersVecFast instruction (the very prompt the vector was
            extracted with) is put before the agent's system prompt: the positive
            instruction for coeff >= 0, the negative one for coeff < 0
    both    both at once

Vectors come from PersVecFast's output directory for one model:
    <vectors_dir>/{trait}.pt                  {layer: vector} for every layer
    <vectors_dir>/{trait}.json                steering sweep summary (best layer)
    <vectors_dir>/vllm_hook/{trait}_L{l}.pt   {"dir": vector}, at the candidate layers
and persona prompts from PersVecFast's trait definitions:
    <traits_dir>/{trait}.json                 {"instruction": {"positive", "negative"}, ...}

Who is steered is set per episode by steering groups (config `steering:`, a
group or a list of groups). A group picks a pool of agents, then a selection
from that pool:
    pool:   every agent, narrowed by `team` (good | evil), `roles` and/or `agents`
            (agent ids or seat indices)
    select: all | one | count (with `count`) | fraction (with `fraction`);
            one / count / fraction are drawn at random each episode
and gives them `trait` (or a weighted mix `traits`), `coefficient` and `layer`
(an int, or "auto" = the trait's best layer in its PersVecFast sweep).
`random_control: true` swaps the vector for a random direction of the same norm.
`roleplay: true` wraps prompt personas in PersVecFast's method-actor framing.
Vector options (defaults as above; logged only when set):
    phase: prefill | decode       steer only while reading the prompt / only the generated tokens (vLLM-Hook
                                  steering.phase; default both)
    method: target                vLLM-Hook adjust_rs: each steered token's projection on the unit vector is set to
                                  a target, h += (target - h . v_hat) v_hat, instead of h += coeff * v. The
                                  coefficient sets the target: m + coeff * |v| / 2, m the projection of the midpoint
                                  between the trait's positive and negative extraction answers (<trait>_raw.pt), so
                                  +1 = the positive answers' mean, -1 = the negative ones', 0 = halfway
    project_out: <trait>          remove the vector's component along that trait's vector at the same layer, then
                                  rescale to the original norm (e.g. careful-reasoning: steer without the part that
                                  points towards quick answers); the random control is projected the same way
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from testbed.logging_.episode_logger import retry_io
from testbed.util import read_json, save_pt

RANDOM_CONTROL_SEED = 0  # as PersVecFast: the control at layer l uses seed RANDOM_CONTROL_SEED + l


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


@dataclass(frozen=True)
class Steer:
    layer: Optional[int]          # None: no vector
    coeff: float
    vector_path: Optional[str]    # absolute path to a {"dir": vector} file
    label: str                    # human-readable, e.g. "evil@L16x2.5", "prompt(evil)"
    persona: Optional[str] = None  # put before the system prompt; None: no prompt persona
    from_turn: int = 0            # unsteered (no vector, no persona) on turns before this one
    phase: str = "both"           # both | prefill | decode (vLLM-Hook steering.phase)
    method: str = "add"           # add (add_vector) | target (adjust_rs; vector_path holds a unit dir + avg_proj)

    def to_log(self) -> dict:
        out = {"label": self.label, "layer": self.layer, "coeff": self.coeff, "vector_path": self.vector_path,
               "persona": self.persona}
        if self.from_turn:
            out["from_turn"] = self.from_turn
        if self.phase != "both":
            out["phase"] = self.phase
        if self.method != "add":
            out["method"] = self.method
        return out

    def active(self, turn: int) -> bool:
        return turn >= self.from_turn


def random_control(vector: torch.Tensor, seed: int) -> torch.Tensor:
    """A random direction with the same norm as `vector` (PersVecFast's steering null baseline)."""
    direction = torch.randn(vector.shape, generator=torch.Generator().manual_seed(seed))
    return direction / direction.norm() * vector.float().norm()


class VectorStore:
    """Persona vectors of one model, read from a PersVecFast output directory."""

    def __init__(self, vectors_dir, cache_dir, traits_dir=None) -> None:
        self.dir = Path(os.path.expandvars(str(vectors_dir)))
        self.cache_dir = Path(cache_dir).resolve()
        # PersVecFast layout: <repo>/out/persona/<model>/bf16 and <repo>/data/persona_vectors
        self.traits_dir = Path(os.path.expandvars(str(traits_dir))) if traits_dir else \
            self.dir.parents[3] / "data" / "persona_vectors"
        self._vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self._summaries: Dict[str, dict] = {}

    def instruction(self, trait: str, positive: bool, roleplay: bool = False) -> str:
        """The trait's PersVecFast system-prompt instruction (the one its vector was extracted with)."""
        path = self.traits_dir / f"{slugify(trait)}.json"
        if not retry_io(path.exists):
            raise FileNotFoundError(f"no trait definition {path}; set `traits_dir` to PersVecFast/data/persona_vectors")
        definition = retry_io(read_json, path)
        if not positive:
            return definition["instruction"]["negative"]
        if roleplay:
            from persvecfast.prompts import positive_instruction
            return positive_instruction(definition, roleplay=True)
        return definition["instruction"]["positive"]

    def traits(self) -> List[str]:
        return sorted(p.stem for p in retry_io(lambda: list(self.dir.glob("*.pt"))) if not p.stem.endswith("_raw"))

    def vectors(self, trait: str) -> Dict[int, torch.Tensor]:
        slug = slugify(trait)
        if slug not in self._vectors:
            path = self.dir / f"{slug}.pt"
            if not retry_io(path.exists):
                raise FileNotFoundError(f"no persona vector {path}; known traits: {self.traits()}")
            loaded = retry_io(torch.load, path, map_location="cpu")
            self._vectors[slug] = {int(l): v.float() for l, v in loaded.items()}
        return self._vectors[slug]

    def num_layers(self) -> int:
        return len(self.vectors(self.traits()[0]))

    def summary(self, trait: str) -> dict:
        # read once per trait: every run config asks again, and each read is a round trip to scratch, which drops out
        slug = slugify(trait)
        if slug not in self._summaries:
            path = self.dir / f"{slug}.json"
            self._summaries[slug] = retry_io(read_json, path) if retry_io(path.exists) else {}
        return self._summaries[slug]

    def candidate_layers(self) -> List[int]:
        """The layers PersVecFast evaluated steering at (the same for every trait of a model)."""
        for trait in self.traits():
            layers = self.summary(trait).get("layers")
            if layers:
                return sorted(int(l) for l in layers)
        raise ValueError(f"no {{trait}}.json with 'layers' in {self.dir}; give monitor/steering layers explicitly")

    def best_layer(self, trait: str) -> int:
        """Layer with the largest steering effect at the strongest coefficient (PersVecFast sweep)."""
        sweep = self.summary(trait).get("sweep_scores") or {}

        def delta(scores):
            scores = {float(a): s for a, s in scores.items() if s is not None}
            return scores[max(scores)] - scores.get(0.0, 0.0) if scores else float("-inf")

        if not sweep:
            raise ValueError(f"{trait}: no sweep_scores in {self.dir / slugify(trait)}.json; set `layer:` explicitly")
        return int(max(sweep, key=lambda l: delta(sweep[l])))

    def midpoint(self, trait: str, layer: int) -> torch.Tensor:
        """Mean of the trait's positive and negative extraction answers' residuals at this layer (<trait>_raw.pt)."""
        path = self.dir / f"{slugify(trait)}_raw.pt"
        if not retry_io(path.exists):
            raise FileNotFoundError(f"method: target needs {path} (PersVecFast's per-answer activations)")
        raw = retry_io(torch.load, path, map_location="cpu")
        pos, neg = raw["positive"][layer]["residual"].float(), raw["negative"][layer]["residual"].float()
        return (pos.mean(0) + neg.mean(0)) / 2

    def steer_file(self, traits: Dict[str, float], layer: int, random_ctrl: bool = False,
                   project_out: Optional[str] = None, target: Optional[float] = None) -> str:
        """Path of a {"dir": vector} file for this trait mix at this layer, writing it if needed. project_out: minus
        the component along that trait's vector, rescaled to the norm before. target: a {"dir": unit vector,
        "avg_proj": midpoint projection + target * |v| / 2} file for vLLM-Hook's adjust_rs."""
        slugs = {slugify(t): float(w) for t, w in traits.items()}
        name = "+".join(f"{s}" if w == 1.0 else f"{s}{w:g}" for s, w in sorted(slugs.items()))
        # project_out: the cached file's name carries a fingerprint of the removed direction, so a rebuilt
        # <project_out>.pt gives a new file instead of reusing one made from the old vector
        fingerprint = ""
        if project_out:
            import hashlib
            u = self.vectors(project_out)[layer].float().contiguous()
            fingerprint = "-" + hashlib.sha1(u.numpy().tobytes()).hexdigest()[:8]
        suffix = ("_random" if random_ctrl else "") + \
                 (f"_minus-{slugify(project_out)}{fingerprint}" if project_out else "") + \
                 (f"_target{target:g}" if target is not None else "")
        if len(slugs) == 1 and next(iter(slugs.values())) == 1.0 and not project_out and target is None:
            exported = self.dir / "vllm_hook" / f"{name}_L{layer}{suffix}.pt"
            if retry_io(exported.exists):
                return str(exported.resolve())
        path = self.cache_dir / f"{name}_L{layer}{suffix}.pt"
        if not path.exists():
            vector = sum(w * self.vectors(s)[layer] for s, w in slugs.items())
            if random_ctrl:
                vector = random_control(vector, RANDOM_CONTROL_SEED + layer)
            if project_out:
                u = self.vectors(project_out)[layer]
                u = u / u.norm()
                norm = vector.norm()
                vector = vector - (vector @ u) * u
                vector = vector / vector.norm() * norm
            if target is None:
                save_pt(path, {"dir": vector.clone()})
            else:
                if len(slugs) != 1:
                    raise ValueError("method: target takes one trait")
                unit = vector / vector.norm()
                mid = float(self.midpoint(next(iter(slugs)), layer) @ unit)
                save_pt(path, {"dir": unit.clone(), "avg_proj": mid + target * float(vector.norm()) / 2})
        return str(path)


@dataclass
class SteerGroup:
    traits: Dict[str, float]
    coefficient: float
    layer: Union[int, str] = "auto"
    select: str = "all"
    fraction: Optional[float] = None
    count: Optional[int] = None
    team: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    agents: List[Union[str, int]] = field(default_factory=list)
    random_control: bool = False
    via: str = "vector"           # vector | prompt | both
    roleplay: bool = False
    from_turn: int = 0            # steer from this turn on (0: every turn; qa_debate: 1 = rounds 2+)
    phase: str = "both"           # both | prefill | decode
    method: str = "add"           # add | target (see the module docstring)
    project_out: Optional[str] = None  # a trait whose direction is removed from the vector

    @classmethod
    def from_dict(cls, d: dict) -> "SteerGroup":
        d = dict(d)
        if "trait" in d:
            d["traits"] = {d.pop("trait"): 1.0}
        traits = d.pop("traits", None)
        if not traits:
            raise ValueError(f"steering group needs `trait` or `traits`: {d}")
        if isinstance(traits, (list, tuple)):
            traits = {t: 1.0 for t in traits}
        if "coefficient" not in d and d.get("via") != "prompt":
            raise ValueError(f"steering group needs `coefficient` (only prompt groups may omit it): {d}")
        group = cls(traits=dict(traits), coefficient=float(d.pop("coefficient", 1.0)), **d)
        if group.via not in ("vector", "prompt", "both"):
            raise ValueError(f"via must be vector | prompt | both, got {group.via!r}")
        if group.via == "prompt" and group.random_control:
            raise ValueError("random_control needs a vector (via: vector | both)")
        if group.select not in ("all", "one", "count", "fraction"):
            raise ValueError(f"select must be all | one | count | fraction, got {group.select!r}")
        if group.select == "fraction" and not 0 < (group.fraction or 0) <= 1:
            raise ValueError("select: fraction needs `fraction` in (0, 1]")
        if group.select == "count" and not (group.count or 0) > 0:
            raise ValueError("select: count needs `count` > 0")
        if group.phase not in ("both", "prefill", "decode"):
            raise ValueError(f"phase must be both | prefill | decode, got {group.phase!r}")
        if group.method not in ("add", "target"):
            raise ValueError(f"method must be add | target, got {group.method!r}")
        if group.via == "prompt" and (group.phase != "both" or group.method != "add" or group.project_out):
            raise ValueError("phase / method / project_out need a vector (via: vector | both)")
        if not isinstance(group.from_turn, int) or group.from_turn < 0:
            raise ValueError(f"from_turn must be an integer >= 0, got {group.from_turn!r}")
        if group.team not in (None, "good", "evil"):
            raise ValueError(f"team must be good | evil, got {group.team!r}")
        return group

    def resolve(self, store: VectorStore) -> Steer:
        mix = "+".join(t if w == 1.0 else f"{w:g}*{t}" for t, w in self.traits.items())
        persona, labels = None, []
        if self.via in ("prompt", "both"):
            # each trait's instruction, positive or negative by the sign of weight x coefficient
            persona = "\n".join(store.instruction(t, w * self.coefficient >= 0, self.roleplay)
                                 for t, w in self.traits.items() if w)
            labels.append(f"prompt({'' if self.coefficient >= 0 else '-'}{mix})")
        if self.via == "prompt":
            return Steer(layer=None, coeff=0.0, vector_path=None, label=labels[0], persona=persona,
                         from_turn=self.from_turn)
        layer = self.layer
        if layer == "auto":
            layer = store.best_layer(max(self.traits, key=lambda t: abs(self.traits[t])))
        layer = int(layer)
        options = [o for o in (self.phase if self.phase != "both" else None,
                               "target" if self.method == "target" else None,
                               f"-{self.project_out}" if self.project_out else None) if o]
        labels.append(f"{'random(' + mix + ')' if self.random_control else mix}@L{layer}x{self.coefficient:g}"
                      + (f"[{','.join(options)}]" if options else ""))
        path = store.steer_file(self.traits, layer, self.random_control, self.project_out,
                                self.coefficient if self.method == "target" else None)
        return Steer(layer=layer, coeff=self.coefficient, label="+".join(labels), persona=persona,
                     vector_path=path, from_turn=self.from_turn, phase=self.phase, method=self.method)

    def pool(self, env) -> List[str]:
        ids = list(env.agent_ids())
        pool = ids
        if self.agents:
            wanted = {ids[a] if isinstance(a, int) else a for a in self.agents}
            pool = [a for a in pool if a in wanted]
        if self.team:
            teams = env.teams()
            if not teams:
                raise ValueError(f"steering by team: {type(env).__name__} has no teams")
            pool = [a for a in pool if teams.get(a) == self.team]
        if self.roles:
            roles = env.roles()
            if not roles:
                raise ValueError(f"steering by role: {type(env).__name__} has no roles")
            wanted = {r.lower() for r in self.roles}
            pool = [a for a in pool if roles.get(a, "").lower() in wanted]
        return pool

    def select_from(self, pool: List[str], rng: random.Random) -> List[str]:
        if self.select == "all":
            return list(pool)
        if self.select == "one":
            n = 1
        elif self.select == "count":
            n = self.count
        else:
            n = round(self.fraction * len(pool))
        return sorted(rng.sample(pool, min(n, len(pool))))


class SteeringPlan:
    """Steering groups with their vectors resolved once; assigns agents per episode."""

    def __init__(self, groups: List[SteerGroup], store: Optional[VectorStore]) -> None:
        self.groups = groups
        self.steers = [g.resolve(store) for g in groups]

    @classmethod
    def from_config(cls, cfg, store: Optional[VectorStore]) -> "SteeringPlan":
        if not cfg:
            return cls([], store)
        groups = [cfg] if isinstance(cfg, dict) else list(cfg)
        return cls([SteerGroup.from_dict(g) for g in groups], store)

    def assign(self, env, seed: int) -> Dict[str, Steer]:
        """{agent_id: Steer} for this episode; call after env.reset() (roles and names are set there)."""
        rng = random.Random(seed)
        out: Dict[str, Steer] = {}
        for group, steer in zip(self.groups, self.steers):
            for agent in group.select_from(group.pool(env), rng):
                if agent in out:
                    raise ValueError(f"{agent} is selected by two steering groups")
                out[agent] = steer
        return out

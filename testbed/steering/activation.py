"""Activation steering: add a steering vector into the residual stream via a hook.

Three modes (controlled by SteeringSpec.mode):

  additive  — unconditionally adds coefficient * vector to the hidden state at
              every token position. This is the standard CAA / ActAdd approach.

  adaptive  — per token, tops up the hidden state's projection onto the vector
              direction toward a target of coefficient * ||vector||, but only if
              the token has not yet reached that target. Implements the same
              formula used in PersVecGen's steering.py (see that repo for derivation).
              Prevents overshooting: a token already at or beyond the target gets
              zero correction.

  rotation  — ORBIT-style norm-preserving rotation (Aparin & Gaintseva 2606.06735;
              Nguyen et al. 2606.22357). Rotates h toward v by θ = coefficient
              radians within the plane spanned by (h, v). Preserves ||h|| exactly,
              so the output logit distribution is not disrupted by norm shift.
              Typical range: 0.05 – 0.5 rad. Note: this is NOT the same scale as
              additive/adaptive coefficients; 0.3 rad ≈ 17° rotation.

Vector files:
  .npy  — plain numpy array, loaded as a 1-D float32 tensor.
  .pt   — either a raw tensor OR a PersVecGen-format dict {layer_int: tensor}.
          When the dict format is detected the layer is selected automatically
          from the companion .json metadata file (highest delta at max coefficient).

Layer resolution:
  If the YAML config contains an explicit "layer" key it is used as-is.
  Otherwise the companion .json is read and the best layer is chosen automatically.
  The int layer index is mapped to a dot-path via "layer_path_template"
  (default "model.layers.{}"), which is correct for Qwen3 and most HF decoders.

Env-var expansion:
  vector_path values are passed through os.path.expandvars(), so YAMLs can
  use ${PERSONA_VECTORS_ROOT}/bf16/opportunistic.pt and the path resolves
  differently on local machines vs. the cluster.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional, Tuple

from testbed.types import SteeringSpec


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------

def make_steering_hook(vector, coefficient: float,
                       mode: str = "adaptive") -> Callable:
    """Return a forward hook that steers a module's output hidden state.

    Handles modules whose output is a plain Tensor or a tuple whose first
    element is the hidden-state Tensor (as in all HF decoder blocks).
    """
    import math
    import torch

    vec = vector if hasattr(vector, "to") else torch.tensor(vector)

    if mode == "adaptive":
        # Port of PersVecGen steering.py adaptive formula:
        #   per token, add only enough to reach coefficient * ||v||,
        #   never overshoot. Positive coefficient only adds; negative only removes.
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            v = vec.to(hidden.dtype).to(hidden.device)
            norm = v.norm()
            if norm == 0:
                return output
            v_hat = v / norm
            target = coefficient * norm
            proj = (hidden * v_hat).sum(dim=-1, keepdim=True)
            correction = target - proj
            correction = (correction.clamp(min=0) if coefficient >= 0
                          else correction.clamp(max=0))
            steered = hidden + correction * v_hat
            return (steered,) + tuple(output[1:]) if isinstance(output, tuple) else steered

    elif mode == "rotation":
        # ORBIT-style norm-preserving rotation in the (h, v) plane.
        # coefficient = θ in radians. Preserves ||h|| at every token.
        _cos_t = math.cos(float(coefficient))
        _sin_t = math.sin(float(coefficient))

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            v = vec.to(hidden.dtype).to(hidden.device)

            v_norm = v.norm()
            if v_norm < 1e-8:
                return output
            v_hat = v / v_norm  # [d]

            h_norm = hidden.norm(dim=-1, keepdim=True)          # [b, s, 1]
            h_hat = hidden / h_norm.clamp(min=1e-8)             # [b, s, d]

            # Gram-Schmidt: component of v̂ perpendicular to ĥ
            dot = (h_hat * v_hat).sum(dim=-1, keepdim=True)     # [b, s, 1]
            v_perp = v_hat - dot * h_hat                        # [b, s, d]
            v_perp_n = v_perp.norm(dim=-1, keepdim=True)        # [b, s, 1]

            v_perp_hat = v_perp / v_perp_n.clamp(min=1e-8)

            # Rotate ĥ by θ toward v̂ within the (h, v) plane
            h_rotated = h_norm * (_cos_t * h_hat + _sin_t * v_perp_hat)

            # Where h ∥ v the perpendicular is undefined — leave h unchanged
            h_new = torch.where(v_perp_n < 1e-6, hidden, h_rotated)

            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new

    else:
        # additive: unconditionally adds coefficient * vector
        def hook(module, inputs, output):
            v = vec.to(
                output[0].dtype if isinstance(output, tuple) else output.dtype
            ).to(
                output[0].device if isinstance(output, tuple) else output.device
            )
            if isinstance(output, tuple):
                return (output[0] + coefficient * v,) + tuple(output[1:])
            return output + coefficient * v

    return hook


# ---------------------------------------------------------------------------
# Layer / metadata helpers
# ---------------------------------------------------------------------------

def _best_layer_from_json(json_path: str,
                           template: str = "model.layers.{}") -> str:
    """Read PersVecGen metadata JSON and return the dot-path of the best layer.

    'Best' is defined as highest (max_coefficient_score - baseline_score).
    """
    with open(json_path) as f:
        meta = json.load(f)
    sweep = meta.get("sweep_scores", {})
    if not sweep:
        raise ValueError(f"No sweep_scores found in {json_path}")

    def _delta(layer_key: str) -> float:
        scores = {float(k): float(v) for k, v in sweep[layer_key].items()}
        baseline = scores.get(0.0, min(scores.values()))
        return max(scores.values()) - baseline

    best_key = max(sweep.keys(), key=_delta)
    return template.format(int(best_key))


def _resolve_layer(cfg: dict) -> str:
    """Return the dot-path layer string for a config dict.

    If 'layer' is already present it is returned unchanged.
    Otherwise the companion .json is read and the best layer is chosen.
    """
    if "layer" in cfg:
        return cfg["layer"]
    vp = os.path.expandvars(cfg.get("vector_path", ""))
    if not vp.endswith(".pt"):
        raise ValueError(
            "Auto-layer selection requires a .pt vector file with a companion "
            f".json metadata file; got {vp!r}. Add 'layer:' to the config "
            "or switch to a PersVecGen .pt file."
        )
    json_path = vp[:-3] + ".json"
    template = cfg.get("layer_path_template", "model.layers.{}")
    return _best_layer_from_json(json_path, template)


# ---------------------------------------------------------------------------
# ActivationSteering
# ---------------------------------------------------------------------------

class ActivationSteering:
    """Per-agent activation steering via residual-stream vector injection.

    per_agent  — maps agent_id -> config dict for per-agent overrides.
    default_config — config dict applied to every agent not in per_agent.
    mode       — "additive" (default) or "adaptive" (soft top-up, no overshoot).

    Config dict keys:
      vector_path           path to .npy or PersVecGen .pt (supports ${ENVVAR})
      coefficient           steering strength α (default 1.0)
      layer                 dot-path module name, e.g. "model.layers.20"
                            (optional: auto-selected from companion .json if absent)
      layer_path_template   format string for auto layer selection
                            (default "model.layers.{}", correct for Qwen3 / Llama)
    """

    def __init__(self, per_agent: Dict[str, Dict],
                 default_config: Optional[Dict] = None,
                 mode: str = "adaptive") -> None:
        self._mode = mode
        self._cache: Dict[str, object] = {}

        # Resolve layers once at init so steering_spec() is a cheap dict lookup.
        self.per_agent: Dict[str, Dict] = {}
        for agent_id, cfg in (per_agent or {}).items():
            resolved = dict(cfg)
            resolved["layer"] = _resolve_layer(resolved)
            self.per_agent[agent_id] = resolved

        if default_config is not None:
            resolved = dict(default_config)
            resolved["layer"] = _resolve_layer(resolved)
            self.default_config: Optional[Dict] = resolved
        else:
            self.default_config = None

    def _cfg_for(self, agent_id) -> Optional[Dict]:
        # TextArena passes integer player IDs (0, 1, …); YAML configs use "player_N" strings.
        # Try the raw key first, then the normalised form, so both conventions work.
        candidates = [agent_id]
        if isinstance(agent_id, int):
            candidates.append(f"player_{agent_id}")
        elif isinstance(agent_id, str) and agent_id.isdigit():
            candidates.append(f"player_{agent_id}")
        for key in candidates:
            if key in self.per_agent:
                return self.per_agent[key]
        return self.default_config

    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        return system_prompt, user_prompt  # activation steering leaves text untouched

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        cfg = self._cfg_for(agent_id)
        if cfg is None:
            return None
        return SteeringSpec(
            method="activation",
            layer=cfg["layer"],
            vector_path=os.path.expandvars(cfg["vector_path"]),
            coefficient=float(cfg.get("coefficient", 1.0)),
            mode=self._mode,
        )

    def load_vector(self, agent_id: str):
        """Load (and cache) the steering vector for agent_id.

        Handles both plain tensors (.npy / flat .pt) and PersVecGen dicts
        {layer_int: tensor}. The correct layer int is extracted from the
        already-resolved dot-path in the config (e.g. "model.layers.20" → 20).
        """
        import torch
        import numpy as np

        if agent_id in self._cache:
            return self._cache[agent_id]

        spec = self.steering_spec(agent_id)
        if spec is None:
            raise KeyError(f"No activation steering configured for {agent_id!r}")

        path = spec.vector_path  # already env-var expanded by steering_spec
        if path.endswith(".npy"):
            vec = torch.tensor(np.load(path), dtype=torch.float32)
        else:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                # PersVecGen format: {layer_int: 1-D tensor}
                layer_int = int(spec.layer.split(".")[-1])
                vec = loaded[layer_int].float()
            else:
                vec = loaded.float()

        self._cache[agent_id] = vec
        return vec

    def on_episode_start(self, env) -> None:
        pass

"""SVD persona steering.

Maps a per-agent persona dict (trait → weight) to an activation injection
vector via the SVD persona basis, then registers forward hooks for each
configured layer.

YAML config schema:
    steering:
      default: svd
      basis_path: ${SVD_BASIS}
      mode: additive          # additive | adaptive | rotation
      per_agent:
        player_0:
          hook: attn          # attn | mlp | both | residual
          layers: [10, 18, 27]
          coefficient: 1.0
          persona:
            agreeable:  1.5
            dominant:  -0.5
            warm:       0.0   # zeros are explicit; merged into 'agreeable' if sim > 0.90
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from testbed.probing.svd_probe import SUBMODULE_SUFFIXES
from testbed.steering.activation import make_steering_hook


class SVDSteering:
    """Per-agent activation steering via the SVD persona space.

    Call apply_hooks(agent_id, model) to get (layer_path, hook_fn) pairs
    ready for TransformersPolicy._HookSession.
    """

    def __init__(
        self,
        basis_path: str,
        per_agent: Dict[str, Dict],
        default_config: Optional[Dict] = None,
        mode: str = "additive",
        layer_path_template: str = "model.layers.{}",
    ) -> None:
        import torch

        self._mode = mode
        self._template = layer_path_template

        basis = torch.load(os.path.expandvars(basis_path),
                           map_location="cpu", weights_only=False)
        self._slugs_dedup: List[str] = basis["slugs"]
        self._merge_map: Dict[str, str] = basis["merge_map"]
        self._slug_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self._slugs_dedup)}
        self._Vk: Dict[int, "torch.Tensor"] = basis["Vk"]   # {layer: [k, d]}
        self._C: Dict[int, "torch.Tensor"] = basis["C"]     # {layer: [N_dedup, k]}

        # Basis/hook consistency guard: per-agent hook must match basis hook
        # (unless the agent requests "both", which is an explicit multi-hook opt-in).
        basis_hook = basis.get("hook")
        normalized_per_agent: Dict = {str(k): v for k, v in (per_agent or {}).items()}
        if basis_hook:
            for agent_id, cfg in normalized_per_agent.items():
                agent_hook = cfg.get("hook")
                if agent_hook and agent_hook != "both" and agent_hook != basis_hook:
                    import warnings
                    warnings.warn(
                        f"Basis was built with hook={basis_hook!r} but agent {agent_id!r} is "
                        f"configured for hook={agent_hook!r}. Consider rebuilding the basis "
                        f"with --hook {agent_hook} or changing the agent hook."
                    )

        # TextArena passes integer player IDs; YAML uses "player_N" strings
        self._per_agent: Dict = normalized_per_agent
        self._default_config = default_config

    def _cfg_for(self, agent_id) -> Optional[Dict]:
        candidates = [str(agent_id)]
        if str(agent_id).isdigit():
            candidates.append(f"player_{agent_id}")
        for key in candidates:
            if key in self._per_agent:
                return self._per_agent[key]
        return self._default_config

    def _build_injection(self, persona: Dict[str, float], layer: int) -> "torch.Tensor":
        """Map persona dict → weight vector w → SVD point z → injection g ∈ ℝ^d.

        z = C[layer].T @ w   (weighted sum of trait coordinates)
        g = Vk[layer].T @ z  (project back to model space)
        """
        import torch
        n = len(self._slugs_dedup)
        w = torch.zeros(n)
        for slug, weight in persona.items():
            canonical = self._merge_map.get(slug, slug)
            idx = self._slug_to_idx.get(canonical)
            if idx is not None:
                w[idx] += weight
        C = self._C[layer]   # [N_dedup, k]
        Vk = self._Vk[layer] # [k, d]
        z = C.T @ w          # [k]
        g = Vk.T @ z         # [d]
        return g

    def apply_hooks(
        self, agent_id, model
    ) -> List[Tuple[str, Callable]]:
        """Return (layer_path, hook_fn) pairs for _HookSession.

        Returns empty list if agent is not configured.
        """
        cfg = self._cfg_for(agent_id)
        if cfg is None:
            return []

        persona = cfg.get("persona", {})
        coefficient = float(cfg.get("coefficient", 1.0))
        hook_type = cfg.get("hook", "attn")
        layers = list(cfg.get("layers", [18]))

        result: List[Tuple[str, Callable]] = []
        for layer in layers:
            g = self._build_injection(persona, layer)
            hook_fn = make_steering_hook(g, coefficient=coefficient, mode=self._mode)
            base = self._template.format(layer)

            if hook_type == "both":
                result.append((base + SUBMODULE_SUFFIXES["attn"], hook_fn))
                result.append((base + SUBMODULE_SUFFIXES["mlp"], hook_fn))
            elif hook_type in SUBMODULE_SUFFIXES:
                result.append((base + SUBMODULE_SUFFIXES[hook_type], hook_fn))
            else:
                raise ValueError(f"Unknown hook type {hook_type!r}")

        return result

    def apply_to_prompt(self, system_prompt: str, user_prompt: str, agent_id: str):
        """SVD steering leaves text untouched; steering is done via activation hooks."""
        return system_prompt, user_prompt

    def steering_spec(self, agent_id: str):
        """SVD path is handled in TransformersPolicy.act() via apply_hooks()."""
        return None

    def on_episode_start(self, env) -> None:
        pass

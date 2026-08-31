"""Direct CAA cosine-similarity probe.

Loads a *_raw.pt file produced by PersVecGen, computes the mean-diff CAA
direction per layer (mean_positive - mean_negative), and hooks the model's
residual stream to return cos(h, direction) at every layer.

No SVD basis required — works directly from the raw extraction files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

_HOOK_KEY = {
    "residual": "residual",
    "attn":     "attn_delta",
    "mlp":      "mlp_delta",
}
_LAYER_SUFFIX = {
    "residual": "",
    "attn":     ".self_attn.o_proj",
    "mlp":      ".mlp.down_proj",
}


class RawCosineProbe:
    """Per-layer cosine similarity between residual stream and CAA direction.

    Usage:
        probe = RawCosineProbe("data/vectors/evil_raw.pt")
        hooks, get_result = probe.make_hook()
        # register hooks, run model forward pass, then:
        scores = get_result()   # {layer_str: {"cos_sim": float}}
        reward = probe.mean_cos_sim(scores)
    """

    def __init__(
        self,
        raw_pt_path: str,
        hook: str = "residual",
        layer_path_template: str = "model.layers.{}",
        layer_start: int = 0,
    ) -> None:
        if hook not in _HOOK_KEY:
            raise ValueError(f"hook must be one of {list(_HOOK_KEY)}; got {hook!r}")

        raw = torch.load(str(raw_pt_path), map_location="cpu", weights_only=False)
        hook_key = _HOOK_KEY[hook]

        self.layers: List[int] = sorted(raw["layers"])
        self.layer_start = layer_start
        self.hook = hook
        self.layer_path_template = layer_path_template

        # Normalised CAA direction per layer: (mean_pos - mean_neg) / ||...||
        self.directions: Dict[int, torch.Tensor] = {}
        for l in self.layers:
            if l not in raw["positive"] or hook_key not in raw["positive"][l]:
                continue
            pos = raw["positive"][l][hook_key].float().mean(0)
            neg = raw["negative"][l][hook_key].float().mean(0)
            d = pos - neg
            self.directions[l] = d / d.norm().clamp(min=1e-8)

        self.layers = [l for l in self.layers if l in self.directions]

    # ------------------------------------------------------------------

    def make_hook(self) -> Tuple[List[Tuple[str, Callable]], Callable]:
        """Return ([(path, hook_fn), ...], get_result_fn) — same contract as SVDPersonaProbe."""
        scores: Dict[int, Optional[float]] = {l: None for l in self.layers}

        def _make_layer_hook(l: int):
            d = self.directions[l]

            def _hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                # average over all token positions, not just last
                mean_h = h[0, :, :].detach().float().mean(0)  # [hidden_dim]
                h_norm = mean_h.norm().clamp(min=1e-8)
                scores[l] = (d.to(mean_h.device) @ mean_h / h_norm).item()

            return _hook

        suffix = _LAYER_SUFFIX[self.hook]
        hooks = [
            (self.layer_path_template.format(l) + suffix, _make_layer_hook(l))
            for l in self.layers
        ]

        def _get_result() -> Dict[str, Dict]:
            return {
                str(l): {"cos_sim": scores[l]}
                for l in self.layers
                if scores[l] is not None
            }

        return hooks, _get_result

    # ------------------------------------------------------------------

    def mean_cos_sim(
        self,
        scores_dict: Dict[str, Dict],
        layer_start: Optional[int] = None,
    ) -> float:
        """Mean cosine similarity across layers >= layer_start."""
        cutoff = layer_start if layer_start is not None else self.layer_start
        vals = [
            v["cos_sim"]
            for l_str, v in scores_dict.items()
            if int(l_str) >= cutoff and v.get("cos_sim") is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

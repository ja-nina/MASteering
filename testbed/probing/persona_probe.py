"""Passive persona probing via residual-stream projections.

Attaches read-only forward hooks to one or more decoder layers during
model.generate() and accumulates running-mean hidden states over the
completion tokens.  Projects each layer's mean state onto all available
PersVecGen trait vectors.

Config (in YAML):
    probing:
      enabled:             true
      vectors_dir:         ${PERSONA_VECTORS_ROOT}/bf16
      layers:              [10, 20, 29]   # probe ALL three simultaneously
      layer_path_template: "model.layers.{}"
      window_tokens:       10
      top_k:               5

Backward compat: a single ``layer: N`` key is treated as ``layers: [N]``.

Output format per completion (stored in episode JSONL under "persona_probe"):
    {
      "10": {"mean": {trait: float, ...}, "chunks": [{token, scores}, ...]},
      "20": {...},
      "29": {...}
    }
"""
from __future__ import annotations

import os
import pathlib
from typing import Callable, Dict, List, Optional, Tuple


class PersonaProbe:
    """Score every agent completion against all persona trait vectors.

    All ``layers`` are probed simultaneously within a single generate() call —
    one forward hook per layer, sharing the same vector set.

    Attributes
    ----------
    layers       list of layer integers being probed
    layer_paths  corresponding dot-paths for hook registration
    n_traits     number of trait vectors loaded
    """

    def __init__(
        self,
        vectors_dir: str,
        layer: int = 20,
        layers: Optional[List[int]] = None,
        layer_path: Optional[str] = None,
        layer_path_template: str = "model.layers.{}",
        window_tokens: int = 10,
        top_k: int = 5,
    ) -> None:
        """
        vectors_dir          directory of <trait>.pt files (env-var expanded)
        layers               list of layer ints to probe (e.g. [10, 20, 29])
        layer                single-layer fallback for backward compat
        layer_path_template  format string for hook registration
        window_tokens        chunk size in generated tokens (0 = global mean only)
        top_k                number of top traits to surface in qualitative output
        """
        if layers is not None:
            self.layers = list(layers)
        else:
            self.layers = [layer]

        self.layer_path_template = layer_path_template
        self.layer_paths = [layer_path_template.format(l) for l in self.layers]

        # Keep single-layer shims for code that checks .layer / .layer_path
        self.layer      = self.layers[0]
        self.layer_path = self.layer_paths[0]

        self.window_tokens = window_tokens
        self.top_k = top_k
        self._vectors = self._load_all(os.path.expandvars(vectors_dir))

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_all(self, vectors_dir: str) -> Dict[str, Tuple["torch.Tensor", float]]:
        """Load all .pt trait vectors; store (v_hat, norm) pairs.

        Vectors are loaded once and shared across all probed layers — each
        layer's hidden state is projected onto the same set of unit vectors.
        The layer index used for loading is the first entry in self.layers.
        """
        import torch

        vecs: Dict[str, Tuple["torch.Tensor", float]] = {}
        for pt_file in sorted(pathlib.Path(vectors_dir).glob("*.pt")):
            trait = pt_file.stem
            loaded = torch.load(
                str(pt_file), map_location="cpu", weights_only=False
            )
            # Use the first probe layer as the load key (all layers share vectors)
            v = loaded[self.layers[0]] if isinstance(loaded, dict) else loaded
            v = v.float()
            norm = v.norm().item()
            if norm > 0:
                vecs[trait] = (v / norm, norm)
        return vecs

    @property
    def n_traits(self) -> int:
        return len(self._vectors)

    # ── Hook factory ───────────────────────────────────────────────────────

    def make_hook(self) -> Tuple[List[Tuple[str, Callable]], Callable]:
        """Return ([(layer_path, hook_fn), ...], get_result_fn).

        Register every (layer_path, hook_fn) pair with _HookSession.
        Call get_result_fn() after generate() to retrieve:

            {
              "10": {"mean": {trait: float}, "chunks": [{token, scores}, ...]},
              "20": {...},
              "29": {...}
            }
        """
        # One independent state dict per layer
        states: Dict[int, Dict] = {
            l: {
                "n": 0,
                "mean": None,
                "chunk_sum": None,
                "chunk_n": 0,
                "chunks": [],
            }
            for l in self.layers
        }
        w = self.window_tokens if self.window_tokens > 0 else None

        def _make_layer_hook(layer_int: int):
            st = states[layer_int]

            def _hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                last = h[:, -1, :].detach().float().mean(dim=0)

                n = st["n"] + 1
                st["n"] = n
                if st["mean"] is None:
                    st["mean"] = last.clone()
                else:
                    st["mean"] = st["mean"] + (last - st["mean"]) / n

                if w is not None:
                    if st["chunk_sum"] is None:
                        st["chunk_sum"] = last.clone()
                        st["chunk_n"] = 1
                    else:
                        st["chunk_sum"] = st["chunk_sum"] + last
                        st["chunk_n"] += 1

                    if st["chunk_n"] >= w:
                        chunk_mean = st["chunk_sum"] / st["chunk_n"]
                        st["chunks"].append({
                            "token": n,
                            "scores": self._project(chunk_mean),
                        })
                        st["chunk_sum"] = None
                        st["chunk_n"] = 0

            return _hook

        hooks = [
            (self.layer_path_template.format(l), _make_layer_hook(l))
            for l in self.layers
        ]

        def _get_result() -> Dict[str, Dict]:
            result = {}
            for l in self.layers:
                st = states[l]
                chunks = list(st["chunks"])
                if st["chunk_sum"] is not None and st["chunk_n"] > 0:
                    chunk_mean = st["chunk_sum"] / st["chunk_n"]
                    chunks.append({
                        "token": st["n"],
                        "scores": self._project(chunk_mean),
                    })
                mean_scores = (
                    self._project(st["mean"]) if st["mean"] is not None else {}
                )
                result[str(l)] = {"mean": mean_scores, "chunks": chunks}
            return result

        return hooks, _get_result

    # ── Projection helper ─────────────────────────────────────────────────

    def _project(self, h: "torch.Tensor") -> Dict[str, float]:
        """Project hidden state onto each trait vector; score = (h·v̂)/‖v‖."""
        return {
            trait: float((h * v_hat.to(h.device)).sum()) / v_norm
            for trait, (v_hat, v_norm) in self._vectors.items()
        }

    # ── Qualitative helpers ────────────────────────────────────────────────

    def top_traits(
        self, scores: Dict[str, float], k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        k = k if k is not None else self.top_k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def qualitative_labels(self, scores: Dict[str, float]) -> List[str]:
        return [t.replace("-", " ") for t, _ in self.top_traits(scores)]

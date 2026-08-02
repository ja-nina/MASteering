"""Passive persona probing via residual-stream projections.

Attaches a read-only forward hook to a decoder layer during model.generate()
and accumulates a running-mean hidden state over the completion tokens.
Projects that mean state onto all available PersVecGen trait vectors to score
which personas the agent's output aligns with.

Design goals
────────────
  • Zero extra inference — the hook piggybacks on generate() that is already
    happening; it only performs dot products.
  • O(1) memory — uses an EMA running mean, never stores per-token tensors.
  • Qualitative output — top-K trait labels per completion, not raw numbers.
  • Adaptable — works for any game; just point vectors_dir at the right precision
    directory for the model being run.

Config (in YAML):
    probing:
      enabled:             true
      vectors_dir:         ${PERSONA_VECTORS_ROOT}/bf16
      layer:               20          # int key inside PersVecGen .pt dicts
      layer_path_template: "model.layers.{}"
      window_tokens:       10          # EMA window size (0 = global mean)
      top_k:               5           # traits reported per step
"""
from __future__ import annotations

import os
import pathlib
from typing import Callable, Dict, List, Optional, Tuple


class PersonaProbe:
    """Score every agent completion against all persona trait vectors.

    Attributes
    ----------
    layer_path   dot-path used to register the hook, e.g. "model.layers.20"
    n_traits     number of trait vectors loaded
    """

    def __init__(
        self,
        vectors_dir: str,
        layer: int = 20,
        layer_path: Optional[str] = None,
        window_tokens: int = 10,
        top_k: int = 5,
    ) -> None:
        """
        vectors_dir     directory of <trait>.pt files (env-var expanded)
        layer           int key inside PersVecGen .pt dicts (e.g. 20)
        layer_path      dot-path for hook registration; defaults to
                        "model.layers.<layer>"
        window_tokens   EMA window size over generated tokens
                        (0 = cumulative mean over full completion)
        top_k           number of top traits to surface in qualitative output
        """
        self.layer = layer
        self.layer_path = layer_path or f"model.layers.{layer}"
        self.window_tokens = window_tokens
        self.top_k = top_k
        self._vectors = self._load_all(os.path.expandvars(vectors_dir))

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_all(self, vectors_dir: str) -> Dict[str, "torch.Tensor"]:
        """Load and pre-normalise all .pt trait vectors at self.layer."""
        import torch

        vecs: Dict[str, "torch.Tensor"] = {}
        for pt_file in sorted(pathlib.Path(vectors_dir).glob("*.pt")):
            trait = pt_file.stem
            loaded = torch.load(
                str(pt_file), map_location="cpu", weights_only=False
            )
            v = loaded[self.layer] if isinstance(loaded, dict) else loaded
            v = v.float()
            norm = v.norm()
            if norm > 0:
                vecs[trait] = v / norm  # pre-normalised: dot product = cosine sim
        return vecs

    @property
    def n_traits(self) -> int:
        return len(self._vectors)

    # ── Hook factory ───────────────────────────────────────────────────────

    def make_hook(self) -> Tuple[Callable, Callable]:
        """Return (hook_fn, get_scores_fn) for one generate() call.

        hook_fn    — forward hook to register on the probe layer
        get_scores_fn — call after generate() to retrieve {trait: projection}

        The hook accumulates a running mean over the last `window_tokens`
        newly-generated token positions using an exponential moving average.
        Pass window_tokens=0 to get a cumulative mean over all tokens.
        """
        state: Dict = {"n": 0, "mean": None}
        w = self.window_tokens

        def _hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Grab the final token position — the just-generated token.
            last = h[:, -1, :].detach().float().mean(dim=0)  # [hidden_dim]
            n = state["n"] + 1
            state["n"] = n
            if state["mean"] is None:
                state["mean"] = last
            elif w == 0 or n <= w:
                # Welford online mean (exact over first w tokens or all tokens)
                state["mean"] = state["mean"] + (last - state["mean"]) / n
            else:
                # EMA: weights older tokens exponentially less
                alpha = 1.0 / w
                state["mean"] = (1.0 - alpha) * state["mean"] + alpha * last

        def _get_scores() -> Dict[str, float]:
            h = state["mean"]
            if h is None:
                return {}
            h_norm = h / (h.norm() + 1e-8)
            return {
                trait: float((h_norm * v_hat.to(h.device)).sum())
                for trait, v_hat in self._vectors.items()
            }

        return _hook, _get_scores

    # ── Qualitative helpers ────────────────────────────────────────────────

    def top_traits(
        self, scores: Dict[str, float], k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Return top-k (trait, score) pairs sorted by projection (descending)."""
        k = k if k is not None else self.top_k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def qualitative_labels(self, scores: Dict[str, float]) -> List[str]:
        """Human-readable trait names for the top-k projections."""
        return [t.replace("-", " ") for t, _ in self.top_traits(scores)]

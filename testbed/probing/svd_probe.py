"""SVD persona-space probe.

Attaches read-only forward hooks to decoder sub-layers during model.generate()
and projects the final-token hidden state onto the SVD persona basis built by
scripts/build_svd_basis.py.

Config (in YAML):
    probing:
      mode:          svd
      basis_path:    ${SVD_BASIS}
      hook:          attn          # attn | mlp | residual
      layers:        [10, 18, 27]
      layer_path_template: "model.layers.{}"
      window_tokens: 10
      top_k:         5

Output per turn (stored in episode JSONL under "persona_probe", the same key used
by PersonaProbe — the exact key depends on how the orchestrator/logger stores
_last_probe):
    {
      "18": {"z": [0.82, -0.31, ...], "top_traits": [["agreeable", 0.71], ...]},
      "27": {...}
    }
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

# Submodule suffix appended to "model.layers.{N}" for each hook type.
# Must match PersVecGen SUBMODULE_PATHS exactly.
SUBMODULE_SUFFIXES: Dict[str, str] = {
    "attn":     ".self_attn.o_proj",
    "mlp":      ".mlp.down_proj",
    "residual": "",
}


class SVDPersonaProbe:
    """Project agent hidden states onto the SVD persona basis.

    Drop-in companion to PersonaProbe; activated when probing.mode == "svd".
    Uses the same make_hook() → (hooks_list, get_result_fn) contract.
    """

    def __init__(
        self,
        basis_path: str,
        layers: Optional[List[int]] = None,
        hook: str = "attn",
        layer_path_template: str = "model.layers.{}",
        window_tokens: int = 10,
        top_k: int = 5,
        layer: int = 18,   # backward-compat fallback; use `layers` instead
    ) -> None:
        import torch

        if hook not in SUBMODULE_SUFFIXES:
            raise ValueError(f"hook must be one of {list(SUBMODULE_SUFFIXES)}; got {hook!r}")

        self.hook = hook
        self.layer_path_template = layer_path_template
        self.window_tokens = window_tokens
        self.top_k = top_k

        basis = torch.load(os.path.expandvars(basis_path),
                           map_location="cpu", weights_only=False)

        if layers is not None:
            self.layers = list(layers)
        else:
            self.layers = sorted(basis["Vk"].keys())  # default: all layers in basis

        # Basis/hook consistency guard
        if basis.get("hook") and basis["hook"] != hook:
            raise ValueError(
                f"Basis was built with hook={basis['hook']!r} but probe is configured "
                f"for hook={hook!r}. Rebuild the basis with --hook {hook} or change probe hook."
            )

        self._slugs: List[str] = basis["slugs"]
        self._Vk: Dict[int, "torch.Tensor"] = basis["Vk"]    # {layer: [k, d]}
        self._C: Dict[int, "torch.Tensor"] = basis["C"]       # {layer: [N_dedup, k]}

    def _layer_path(self, layer_int: int) -> str:
        base = self.layer_path_template.format(layer_int)
        return base + SUBMODULE_SUFFIXES[self.hook]

    def make_hook(self) -> Tuple[List[Tuple[str, Callable]], Callable]:
        """Return ([(path, hook_fn), ...], get_result_fn).

        Compatible with PersonaProbe.make_hook() contract.
        Register every pair via _HookSession; call get_result_fn() after
        generate() to retrieve the SVD coordinates for each layer.
        """
        import torch

        states: Dict[int, Dict] = {
            l: {"n": 0, "mean_z": None,
                "chunk_sum_z": None, "chunk_n": 0, "chunks": []}
            for l in self.layers
        }
        w = self.window_tokens if self.window_tokens > 0 else None

        def _make_layer_hook(layer_int: int):
            st = states[layer_int]
            Vk = self._Vk[layer_int]  # [k, d]

            def _hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                last = h[:, -1, :].detach().float().mean(dim=0)  # [d]
                z = Vk.to(last.device) @ last                    # [k]

                n = st["n"] + 1
                st["n"] = n
                if st["mean_z"] is None:
                    st["mean_z"] = z.clone()
                else:
                    st["mean_z"] = st["mean_z"] + (z - st["mean_z"]) / n

                if w is not None:
                    if st["chunk_sum_z"] is None:
                        st["chunk_sum_z"] = z.clone()
                        st["chunk_n"] = 1
                    else:
                        st["chunk_sum_z"] = st["chunk_sum_z"] + z
                        st["chunk_n"] += 1
                    if st["chunk_n"] >= w:
                        chunk_z = st["chunk_sum_z"] / st["chunk_n"]
                        st["chunks"].append({
                            "token": n,
                            "z": chunk_z.tolist(),
                        })
                        st["chunk_sum_z"] = None
                        st["chunk_n"] = 0

            return _hook

        hooks = [
            (self._layer_path(l), _make_layer_hook(l))
            for l in self.layers
        ]

        def _get_result() -> Dict[str, Dict]:
            result = {}
            for l in self.layers:
                st = states[l]
                chunks = list(st["chunks"])
                if st["chunk_sum_z"] is not None and st["chunk_n"] > 0:
                    chunk_z = st["chunk_sum_z"] / st["chunk_n"]
                    chunks.append({"token": st["n"], "z": chunk_z.tolist()})
                mean_z = st["mean_z"]
                if mean_z is None:
                    top_traits: List[List] = []
                    z_list: List[float] = []
                else:
                    top_traits = self._nearest_traits(mean_z, l)
                    z_list = mean_z.tolist()
                result[str(l)] = {"z": z_list, "top_traits": top_traits, "chunks": chunks}
            return result

        return hooks, _get_result

    def rank_traits(self, z_list: List[float], layer: int) -> List[List]:
        """Public: rank traits for a raw z-vector (list of floats) at a given layer."""
        import torch
        z = torch.tensor(z_list, dtype=torch.float32)
        return self._nearest_traits(z, layer)

    def _nearest_traits(self, z: "torch.Tensor", layer: int) -> List[List]:
        """Return top-k [[slug, cosine_sim], ...] for z against C[layer]."""
        C = self._C[layer].to(z.device)  # [N_dedup, k]
        z_norm = z.norm().clamp(min=1e-8)
        C_norms = C.norm(dim=1).clamp(min=1e-8)   # [N_dedup]
        sims = (C @ z) / (C_norms * z_norm)       # [N_dedup]
        k = min(self.top_k, len(self._slugs))
        topk_vals, topk_idx = sims.topk(k)
        return [[self._slugs[i.item()], v.item()] for v, i in zip(topk_vals, topk_idx)]

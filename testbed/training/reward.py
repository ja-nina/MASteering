"""Reward functions for LoRA training."""
import torch
from typing import Dict, List, Optional


class PersonaReward:
    """Mean cosine-similarity between probe z-vectors and the target direction,
    averaged over all residual-stream layers >= layer_start.

    probe_scores format (from SVDPersonaProbe.make_hook / probe_text):
        { "10": {"z": [...], ...}, "11": {...}, ..., "35": {...} }

    reward = mean_{l >= layer_start} cosine_sim( z_l , target_l )
    """

    def __init__(
        self,
        basis_path: str,
        target_traits: Dict[str, float],
        layer_start: int = 10,
        sign: float = +1.0,
    ):
        import os
        basis = torch.load(os.path.expandvars(basis_path),
                           map_location="cpu", weights_only=False)
        slugs: List[str] = basis["slugs"]
        slug_to_idx = {s: i for i, s in enumerate(slugs)}
        merge_map = basis.get("merge_map", {})
        self.layer_start = layer_start
        self.sign = sign

        w = torch.zeros(len(slugs))
        found, missing = [], []
        for trait, weight in target_traits.items():
            canonical = merge_map.get(trait, trait)
            if canonical in slug_to_idx:
                w[slug_to_idx[canonical]] += weight
                found.append(trait)
            else:
                missing.append(trait)

        if missing:
            print(f"[PersonaReward] WARNING: traits not in basis and will be ignored: {missing}")
            print(f"[PersonaReward] Available slugs: {slugs[:20]}{'...' if len(slugs)>20 else ''}")
        if found:
            print(f"[PersonaReward] Targeting traits: {found}")
        else:
            print("[PersonaReward] ERROR: no target traits found — reward will be 0 everywhere!")

        # Direct cosine-sim mode (basis has M_dedup: raw CAA directions in d-space).
        # Probe z is [N_traits] cosine sims → reward = weighted sum, normalised by ||w||.
        # Falls back to SVD z-space mode for old basis files without M_dedup.
        if "M_dedup" in basis:
            self._use_direct = True
            w_norm = w.norm().clamp(min=1e-8)
            self._w_normalized = w / w_norm   # [N_traits] — dot with cos_sim vector
            # Which layers to expect (same as probe)
            self._valid_layers = set(
                l for l in basis["M_dedup"].keys() if l >= layer_start
            )
            print(f"[PersonaReward] mode=direct cosine-sim  layers>={layer_start}: "
                  f"{len(self._valid_layers)} layers")
        else:
            self._use_direct = False
            C_all: Dict[int, torch.Tensor] = basis["C"]   # {layer: [N_dedup, k]}
            self.z_targets: Dict[int, torch.Tensor] = {}
            self.z_target_norms: Dict[int, torch.Tensor] = {}
            for layer, C in C_all.items():
                if layer < layer_start:
                    continue
                C = C.float()
                z_t = C.T @ w
                self.z_targets[layer] = z_t
                self.z_target_norms[layer] = z_t.norm().clamp(min=1e-8)
            n_layers = len(self.z_targets)
            print(f"[PersonaReward] mode=SVD z-space  layers>={layer_start}: {n_layers} layers")

    def __call__(self, probe_scores: Dict[str, dict]) -> float:
        """Score a probe_scores dict (layer_str → {z: [...]}) against the target."""
        total = 0.0
        count = 0
        self.last_cos_sims: Dict[int, float] = {}   # layer → raw cosine sim (pre-sign)
        for layer_key, ld in probe_scores.items():
            layer = int(layer_key)
            if layer < self.layer_start:
                continue
            z_list = ld.get("z") if isinstance(ld, dict) else None
            if not z_list:
                continue
            z = torch.tensor(z_list, dtype=torch.float32)
            if self._use_direct:
                # z is [N_traits] cosine sims; reward = weighted sum
                cos_sim = (z @ self._w_normalized.to(z.device)).item()
            else:
                if layer not in self.z_targets:
                    continue
                z_norm = z.norm().clamp(min=1e-8)
                cos_sim = (z @ self.z_targets[layer]) / (z_norm * self.z_target_norms[layer])
                cos_sim = cos_sim.item()
            self.last_cos_sims[layer] = cos_sim
            total += cos_sim
            count += 1
        if count == 0:
            self.last_mean_cos_sim = 0.0
            return 0.0
        self.last_mean_cos_sim = total / count   # unscaled, pre-sign
        return self.sign * total / count

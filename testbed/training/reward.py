"""Reward functions for LoRA training."""
import torch
from typing import Dict, List


class PersonaReward:
    """Cosine similarity between probe z and a weighted target direction in SVD space.

    ``sign=+1.0`` rewards moving TOWARD the target traits (maximiser adapter).
    ``sign=-1.0`` rewards moving AWAY  from the target traits (minimiser adapter).
    """

    def __init__(self, basis_path: str, layer: int, target_traits: Dict[str, float],
                 sign: float = +1.0):
        import os
        basis = torch.load(os.path.expandvars(basis_path), map_location="cpu", weights_only=False)
        C = basis["C"][layer].float()          # [N_dedup, k]
        slugs = basis["slugs"]
        slug_to_idx = {s: i for i, s in enumerate(slugs)}

        # Build target z-direction: weighted sum of trait coordinates.
        w = torch.zeros(len(slugs))
        merge_map = basis.get("merge_map", {})
        for trait, weight in target_traits.items():
            canonical = merge_map.get(trait, trait)
            if canonical in slug_to_idx:
                w[slug_to_idx[canonical]] += weight
        # z_target = C.T @ w  →  [k]
        self.z_target = C.T @ w
        self.z_target_norm = self.z_target.norm().clamp(min=1e-8)
        self.sign = sign

    def __call__(self, z_list: List[float]) -> float:
        """Score a z-vector (from probe output) against the target direction."""
        z = torch.tensor(z_list, dtype=torch.float32)
        z_norm = z.norm().clamp(min=1e-8)
        cos_sim = (z @ self.z_target) / (z_norm * self.z_target_norm)
        return (self.sign * cos_sim).item()

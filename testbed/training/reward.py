"""Reward functions for LoRA training."""
import torch
from typing import Dict, List, Optional


class PersonaReward:
    """Cosine similarity between probe z and a weighted target direction in SVD space."""

    def __init__(self, basis_path: str, layer: int, target_traits: Dict[str, float]):
        import os
        basis = torch.load(os.path.expandvars(basis_path), map_location="cpu", weights_only=False)
        C = basis["C"][layer].float()          # [N_dedup, k]
        slugs = basis["slugs"]
        slug_to_idx = {s: i for i, s in enumerate(slugs)}

        # Build target z-direction: weighted sum of trait coordinates
        w = torch.zeros(len(slugs))
        merge_map = basis.get("merge_map", {})
        for trait, weight in target_traits.items():
            canonical = merge_map.get(trait, trait)
            if canonical in slug_to_idx:
                w[slug_to_idx[canonical]] += weight
        # z_target = C.T @ w  [k]
        self.z_target = (C.T @ w)
        self.z_target_norm = self.z_target.norm().clamp(min=1e-8)

    def __call__(self, z_list: List[float]) -> float:
        """Score a z-vector (from probe output) against the target direction."""
        z = torch.tensor(z_list, dtype=torch.float32)
        z_norm = z.norm().clamp(min=1e-8)
        return ((z @ self.z_target) / (z_norm * self.z_target_norm)).item()


class TaskReward:
    """Win/loss reward from game outcome."""

    def __call__(self, game_rewards: Dict[int, float], player_id: int) -> float:
        return float(game_rewards.get(player_id, 0.0))


class CombinedReward:
    def __init__(self, persona_reward: PersonaReward, task_reward: TaskReward,
                 persona_weight: float = 1.0, task_weight: float = 1.0):
        self.persona = persona_reward
        self.task = task_reward
        self.persona_weight = persona_weight
        self.task_weight = task_weight

    def __call__(self, z_list: Optional[List[float]], game_rewards: Dict[int, float],
                 player_id: int) -> float:
        r = 0.0
        if z_list and self.persona_weight:
            r += self.persona_weight * self.persona(z_list)
        if self.task_weight:
            r += self.task_weight * self.task(game_rewards, player_id)
        return r

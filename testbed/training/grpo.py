"""GRPO (Group Relative Policy Optimization) for dual persona LoRA training.

Collects K rollouts per update step (a "group"), normalises rewards within
the group to form advantages, and does one gradient step per adapter.

Reference: DeepSeek-R1 (Shao et al. 2024) — same algorithm, adapted for
multi-turn self-play with a dense SVD cosine-similarity reward.

Advantage formula:
    a_i = (mean_reward_i - μ_group) / (σ_group + ε)

where mean_reward_i is the mean per-turn reward across episode i.
All turns within episode i share the same advantage a_i.
"""
from __future__ import annotations

from typing import List, Optional

import torch


def grpo_step(
    episode_records: List[List],      # [K][T] — K episodes, each a list of TurnRecords
    episode_rewards: List[List[float]],  # [K][T] — matching per-turn rewards
    optimizer,
    max_grad_norm: float = 1.0,
    retain_graph: bool = False,
    eps: float = 1e-8,
) -> float:
    """One GRPO gradient step over K parallel rollouts.

    Returns the scalar loss value (detached).

    Args:
        retain_graph: pass True for the FIRST of two adapter backward passes
            so the computation graph survives for the second adapter's step.
    """
    if not episode_records:
        return 0.0

    # Episode-level mean reward (scalar per episode)
    ep_means: List[float] = [
        sum(rs) / max(len(rs), 1) for rs in episode_rewards
    ]

    # Group statistics
    K = len(ep_means)
    group_mean = sum(ep_means) / K
    group_var = sum((m - group_mean) ** 2 for m in ep_means) / max(K, 1)
    group_std = group_var ** 0.5

    # Build loss: -log_prob * advantage, summed over all turns in all episodes
    loss = torch.tensor(0.0)
    for records, ep_mean in zip(episode_records, ep_means):
        advantage = (ep_mean - group_mean) / (group_std + eps)
        for record in records:
            if record.log_prob is not None:
                loss = loss + (-record.log_prob * float(advantage))

    if loss.requires_grad:
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(
            [p for pg in optimizer.param_groups for p in pg["params"]],
            max_grad_norm,
        )
        optimizer.step()

    return loss.item()


def group_stats(episode_rewards: List[List[float]]) -> dict:
    """Compute summary statistics over a group of K episodes (for logging)."""
    ep_means = [sum(rs) / max(len(rs), 1) for rs in episode_rewards]
    K = len(ep_means)
    if K == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = sum(ep_means) / K
    sigma = (sum((m - mu) ** 2 for m in ep_means) / K) ** 0.5
    return {"mean": mu, "std": sigma, "min": min(ep_means), "max": max(ep_means)}

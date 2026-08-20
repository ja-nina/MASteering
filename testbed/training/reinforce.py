"""REINFORCE with running baseline."""
from __future__ import annotations
from typing import List, Tuple
import torch


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_step(
    records,             # List[TurnRecord]
    rewards: List[float],
    optimizer,
    baseline: float,
    gamma: float = 0.99,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """One REINFORCE gradient step. Returns (loss_value, updated_baseline)."""
    returns = compute_returns(rewards, gamma)
    mean_r = sum(rewards) / max(len(rewards), 1)
    updated_baseline = 0.9 * baseline + 0.1 * mean_r

    if not records:
        return 0.0, updated_baseline

    ret_tensor = torch.tensor(returns, dtype=torch.float32)
    advantages = (ret_tensor - baseline) / (ret_tensor.std().clamp(min=1e-8) + 1e-8)

    loss = torch.tensor(0.0)
    for record, adv in zip(records, advantages):
        if record.log_prob is not None:
            loss = loss + (-record.log_prob * adv.detach())

    if loss.requires_grad:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for pg in optimizer.param_groups for p in pg["params"]],
            max_grad_norm
        )
        optimizer.step()

    return loss.item(), updated_baseline

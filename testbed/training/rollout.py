"""Episode rollout for self-play training.

Generation is fully decoupled from gradient computation:

  collect_episode()  — pure torch.no_grad(); stores (obs, action, full_ids,
                        probe_z) but no live computation graph.
  recompute_logprobs() — single batched teacher-forcing pass WITH gradients,
                         called once per GRPO group just before backward.

This keeps collection fast (no graph accumulation) and lets the optimizer
step work on freshly computed log_probs without 80 stale graphs in VRAM.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

OPPONENT_ID = 0  # player 0 = frozen opponent
TRAINEE_ID  = 1  # player 1 = model being trained


@dataclass
class TurnRecord:
    obs: str
    action: str
    full_ids: "torch.Tensor"               # [1, input_len+gen_len] cpu tensor
    input_len: int                         # number of prompt tokens in full_ids
    log_prob: Optional["torch.Tensor"]     # filled in by recompute_logprobs()
    probe_z: Optional[List[float]]         # z-vector at reward layer (for reward fn)
    probe_z_all: Optional[Dict[str, List[float]]] = None  # z per layer (for logging)


@dataclass
class Episode:
    records: List[TurnRecord] = field(default_factory=list)
    game_rewards: Dict[int, float] = field(default_factory=dict)


def collect_episode(
    game_id: str,
    num_players: int,
    trainee_policy,
    opponent_policy,
    probe_layer: int,
    reward_fn=None,           # unused; kept for API compat
    system_prompt: str = "You are a strategic game player. Respond concisely.",
    max_turns: int = 50,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Episode:
    """Collect one episode with pure torch.no_grad() — no graphs stored."""
    import textarena as ta
    env = ta.make(game_id)
    env.reset(num_players=num_players)
    episode = Episode()
    turn_count = 0

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()
        if player_id == TRAINEE_ID:
            if verbose:
                print(f"    turn {turn_count+1} trainee...", flush=True)
            t0 = time.time()
            # Generate action — pure no_grad, no computation graph stored.
            action, (full_ids, input_len) = trainee_policy.act(
                system_prompt=system_prompt,
                user_prompt=obs_str,
                agent_id=str(player_id),
                steering=None,
                return_full_ids=True,
            )
            elapsed = time.time() - t0
            probe_z = None
            probe_z_all = None
            if trainee_policy._last_probe:
                layer_data = trainee_policy._last_probe.get(str(probe_layer), {})
                probe_z = layer_data.get("z") or None
                probe_z_all = {
                    k: v["z"] for k, v in trainee_policy._last_probe.items()
                    if v.get("z")
                }
            if verbose:
                words = len(action.split())
                print(f"    turn {turn_count+1} done — {words} words  {elapsed:.1f}s",
                      flush=True)
            episode.records.append(
                TurnRecord(obs_str, action, full_ids, input_len,
                           log_prob=None, probe_z=probe_z, probe_z_all=probe_z_all)
            )
        else:
            if verbose:
                print(f"    turn {turn_count+1} opponent...", flush=True)
            t0 = time.time()
            action, _ = opponent_policy.act(
                system_prompt=system_prompt,
                user_prompt=obs_str,
                agent_id=str(player_id),
                steering=None,
            )
            if verbose:
                print(f"    turn {turn_count+1} done  {time.time()-t0:.1f}s", flush=True)
        done, _ = env.step(action)
        turn_count += 1
        if done:
            break

    game_rewards, _ = env.close()
    episode.game_rewards = {int(k): float(v) for k, v in game_rewards.items()}
    return episode


def recompute_logprobs(episodes: List[Episode], model, device: str) -> None:
    """Fill in record.log_prob for every trainee turn across all episodes.

    Runs one teacher-forcing forward pass per turn WITH gradient tracking.
    Called once after all K episodes are collected, so graphs are created
    fresh right before the backward — not held across the entire collection.
    """
    import torch.nn.functional as F

    for episode in episodes:
        for record in episode.records:
            if record.full_ids is None:
                record.log_prob = None
                continue
            full_ids = record.full_ids.to(device)   # [1, T]
            input_len = record.input_len
            gen_len = full_ids.shape[1] - input_len
            if gen_len <= 0:
                record.log_prob = torch.tensor(0.0)
                continue
            # Forward pass WITH gradients — graph lives only until backward().
            logits = model(full_ids).logits          # [1, T, vocab]
            gen_logits = logits[0, input_len - 1: input_len - 1 + gen_len, :]
            log_probs_matrix = F.log_softmax(gen_logits, dim=-1)
            gen_ids = full_ids[0, input_len:]
            record.log_prob = log_probs_matrix[
                torch.arange(gen_len, device=device), gen_ids
            ].sum()

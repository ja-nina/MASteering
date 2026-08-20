"""Episode rollout for self-play training.

Reward design: the trainee is rewarded based on the OPPONENT's hidden-state
persona projection after reading the trainee's message — i.e., does the
trainee's text cause the opponent to internally represent the target traits?
This is captured by reading trainee_policy._last_probe right after the
opponent's turn (FrozenOpponent calls through to trainee_policy.act(), which
updates _last_probe with the opponent's forward-pass activations).

Generation is decoupled from gradient computation:
  collect_episode()     — pure torch.no_grad(); stores full_ids+input_len
                          but no live computation graph.
  recompute_logprobs()  — teacher-forcing pass WITH gradients, called once
                          per GRPO group just before backward.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

OPPONENT_ID = 0
TRAINEE_ID  = 1


@dataclass
class TurnRecord:
    obs: str
    action: str
    full_ids: "torch.Tensor"               # [1, input_len+gen_len] cpu, no grad
    input_len: int                         # prompt token count in full_ids
    log_prob: Optional["torch.Tensor"]     # filled by recompute_logprobs()
    # Trainee's own hidden-state projection (for logging/analysis only).
    probe_z: Optional[List[float]]
    # Opponent's hidden-state projection after reading this trainee message
    # (the reward signal — did my words shift the opponent's persona?).
    probe_z_opponent: Optional[List[float]] = None
    probe_z_all: Optional[Dict[str, List[float]]] = None


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
    reward_fn=None,
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
    pending_trainee_record: Optional[TurnRecord] = None  # waiting for opponent reaction

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()

        if player_id == TRAINEE_ID:
            if verbose:
                print(f"    turn {turn_count+1} trainee...", flush=True)
            t0 = time.time()
            action, (full_ids, input_len) = trainee_policy.act(
                system_prompt=system_prompt,
                user_prompt=obs_str,
                agent_id=str(player_id),
                steering=None,
                return_full_ids=True,
            )
            elapsed = time.time() - t0

            # Trainee's own probe (kept for logging/analysis only).
            probe_z = None
            probe_z_all = None
            if trainee_policy._last_probe:
                ld = trainee_policy._last_probe.get(str(probe_layer), {})
                probe_z = ld.get("z") or None
                probe_z_all = {
                    k: v["z"] for k, v in trainee_policy._last_probe.items()
                    if v.get("z")
                }
            if verbose:
                print(f"    turn {turn_count+1} done — "
                      f"{len(action.split())} words  {elapsed:.1f}s", flush=True)

            record = TurnRecord(obs_str, action, full_ids, input_len,
                                log_prob=None, probe_z=probe_z,
                                probe_z_all=probe_z_all)
            episode.records.append(record)
            pending_trainee_record = record  # opponent reaction goes here

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
            elapsed = time.time() - t0

            # Capture the opponent's probe — this is the reward signal.
            # _FrozenOpponent calls through to trainee_policy.act(), so
            # _last_probe now holds the opponent's hidden-state activation.
            if pending_trainee_record is not None and trainee_policy._last_probe:
                ld = trainee_policy._last_probe.get(str(probe_layer), {})
                opp_z = ld.get("z") or None
                pending_trainee_record.probe_z_opponent = opp_z
                pending_trainee_record = None

            if verbose:
                print(f"    turn {turn_count+1} done  {elapsed:.1f}s", flush=True)

        done, _ = env.step(action)
        turn_count += 1
        if done:
            break

    game_rewards, _ = env.close()
    episode.game_rewards = {int(k): float(v) for k, v in game_rewards.items()}
    return episode


def recompute_logprobs(episodes: List[Episode], model, device: str) -> None:
    """Fill record.log_prob for every trainee turn, right before backward.

    One teacher-forcing forward pass per turn WITH gradient tracking.
    Graphs are fresh and freed episode-by-episode during grpo_step.
    """
    import torch.nn.functional as F

    for episode in episodes:
        for record in episode.records:
            if record.full_ids is None:
                record.log_prob = None
                continue
            full_ids = record.full_ids.to(device)
            gen_len = full_ids.shape[1] - record.input_len
            if gen_len <= 0:
                record.log_prob = torch.tensor(0.0)
                continue
            logits = model(full_ids).logits                       # [1, T, vocab]
            gen_logits = logits[0,
                                record.input_len - 1:
                                record.input_len - 1 + gen_len, :]
            log_probs_matrix = F.log_softmax(gen_logits, dim=-1)
            gen_ids = full_ids[0, record.input_len:]
            record.log_prob = log_probs_matrix[
                torch.arange(gen_len, device=device), gen_ids
            ].sum()

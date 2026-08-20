"""Episode rollout for self-play training.

Reward design: at each trainee turn, K candidate responses are sampled from
the policy (all conditioned on the SAME observation).  The opponent's hidden-
state probe is run on each candidate via a lightweight forward pass (no
generation required), yielding K cosine-similarity rewards.  These K rewards
are normalised within the turn group to form per-turn GRPO advantages.

Generation is decoupled from gradient computation:
  collect_episode()     — pure torch.no_grad(); stores full_ids+input_len for
                          every candidate; probes opponent on each candidate.
  recompute_logprobs()  — teacher-forcing pass WITH gradients, called once
                          per GRPO step just before backward.
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
    probe_z: Optional[List[float]]         # trainee's own hidden state (logging)
    probe_z_opponent: Optional[List[float]] = None   # opponent reaction (reward)
    probe_z_all: Optional[Dict[str, List[float]]] = None


@dataclass
class TurnGroup:
    """K candidate responses for one trainee turn — the GRPO comparison unit.

    All K records share the same obs; rewards[k] is the opponent's cosine-sim
    reaction to records[k].action.  Advantages are filled by grpo_step.
    """
    obs: str
    records: List[TurnRecord]              # K records, same obs different action
    rewards: List[float]                   # K rewards (one per candidate)
    advantages: Optional[List[float]] = None  # filled by grpo_step


@dataclass
class Episode:
    turn_groups: List[TurnGroup] = field(default_factory=list)
    game_rewards: Dict[int, float] = field(default_factory=dict)


def collect_episode(
    game_id: str,
    num_players: int,
    trainee_policy,
    opponent_policy,
    probe_layer: int,                      # display layer for probe_z_opponent logging
    reward_fn,                             # callable(probe_scores_dict) -> float
    grpo_k: int = 4,
    probe=None,                            # SVDPersonaProbe for per-turn trait display
    system_prompt: str = "You are a strategic game player. Respond concisely.",
    max_turns: int = 50,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Episode:
    """Collect one episode with K candidate responses per trainee turn.

    For each trainee turn:
      1. Generate K different responses (all conditioned on the same obs).
      2. For each response, probe the opponent's hidden state via a single
         forward pass on that response text (no generation needed).
      3. Compute K rewards from the K probe vectors.
      4. Advance the game with response 0.
    """
    import textarena as ta
    env = ta.make(game_id)
    env.reset(num_players=num_players)
    episode = Episode()
    turn_count = 0

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()

        if player_id == TRAINEE_ID:
            if verbose:
                print(f"    turn {turn_count+1} trainee (x{grpo_k})...", flush=True)
            t0 = time.time()

            records: List[TurnRecord] = []
            rewards: List[float] = []

            for k in range(grpo_k):
                action, (full_ids, input_len) = trainee_policy.act(
                    system_prompt=system_prompt,
                    user_prompt=obs_str,
                    agent_id=str(player_id),
                    steering=None,
                    return_full_ids=True,
                )

                # Trainee's own probe (logging only; captured by act())
                probe_z = None
                probe_z_all = None
                if k == 0 and trainee_policy._last_probe:
                    ld = trainee_policy._last_probe.get(str(probe_layer), {})
                    probe_z = ld.get("z") or None
                    probe_z_all = {
                        lk: v["z"]
                        for lk, v in trainee_policy._last_probe.items()
                        if v.get("z")
                    }

                # Opponent's reaction to this specific response (reward signal).
                # probe_text() runs a single forward pass on the action text with
                # the probe hooks active and adapters disabled — no generation.
                # reward_fn takes the full per-layer scores dict and averages
                # cosine-sim over all layers >= reward_layer_start.
                opp_scores = trainee_policy.probe_text(action)
                ld_opp = opp_scores.get(str(probe_layer), {})
                opp_z = ld_opp.get("z") or None   # display-layer z for logging
                reward = reward_fn(opp_scores) if opp_scores else 0.0

                records.append(TurnRecord(
                    obs=obs_str,
                    action=action,
                    full_ids=full_ids,
                    input_len=input_len,
                    log_prob=None,
                    probe_z=probe_z,
                    probe_z_opponent=opp_z,
                    probe_z_all=probe_z_all,
                ))
                rewards.append(reward)

            elapsed = time.time() - t0
            best_r = max(rewards)
            mean_r = sum(rewards) / len(rewards)
            if verbose:
                words = len(records[0].action.split())
                trait_str = ""
                if probe is not None:
                    best_k = rewards.index(best_r)
                    opp_z = records[best_k].probe_z_opponent
                    if opp_z:
                        try:
                            top = probe.rank_traits(opp_z, probe_layer)
                            if top:
                                trait_str = f"  top={top[0][0]}:{top[0][1]:+.2f}"
                        except Exception:
                            pass
                print(
                    f"    turn {turn_count+1} done — {words} words  {elapsed:.1f}s  "
                    f"r=[{mean_r:+.3f} mean, {best_r:+.3f} best]{trait_str}",
                    flush=True,
                )

            episode.turn_groups.append(TurnGroup(obs=obs_str, records=records,
                                                  rewards=rewards))
            # Advance the game with the first candidate response.
            done, _ = env.step(records[0].action)

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
            if verbose:
                print(f"    turn {turn_count+1} done  {elapsed:.1f}s", flush=True)
            done, _ = env.step(action)

        turn_count += 1
        if done:
            break

    game_rewards, _ = env.close()
    episode.game_rewards = {int(k): float(v) for k, v in game_rewards.items()}
    return episode


def recompute_logprobs(episode: Episode, model, device: str) -> None:
    """Fill record.log_prob for every candidate in every turn group.

    Teacher-forcing forward pass WITH gradient tracking.  Called once per
    GRPO step right before backward.  Graphs are freed turn-group-by-turn-
    group during grpo_step.
    """
    import torch.nn.functional as F

    for tg in episode.turn_groups:
        for record in tg.records:
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

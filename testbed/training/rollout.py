"""Episode rollout for self-play training."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# textarena is imported lazily inside collect_episode() to avoid import-time
# failures on platforms where textarena's optional curses dep is missing.

OPPONENT_ID = 0  # player 0 = frozen opponent
TRAINEE_ID  = 1  # player 1 = model being trained


@dataclass
class TurnRecord:
    obs: str
    action: str
    log_prob: "torch.Tensor"   # scalar tensor with grad
    probe_z: Optional[List[float]]  # z-vector at training layer, or None


@dataclass
class Episode:
    records: List[TurnRecord] = field(default_factory=list)
    game_rewards: Dict[int, float] = field(default_factory=dict)


def collect_episode(
    game_id: str,
    num_players: int,
    trainee_policy,                # TransformersPolicy (LoRA model, grad enabled)
    opponent_policy,               # TransformersPolicy or compatible (frozen base)
    probe_layer: int,
    reward_fn=None,                # deprecated / unused; kept for API compatibility
    system_prompt: str = "You are a strategic game player. Respond concisely.",
    max_turns: int = 50,
    seed: Optional[int] = None,
) -> Episode:
    import textarena as ta
    env = ta.make(game_id)
    env.reset(num_players=num_players)
    episode = Episode()
    turn_count = 0

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()
        if player_id == TRAINEE_ID:
            action, log_prob = trainee_policy.act(
                system_prompt=system_prompt,
                user_prompt=obs_str,
                agent_id=str(player_id),
                steering=None,
                return_logprob=True,
            )
            probe_z = None
            if trainee_policy._last_probe:
                layer_data = trainee_policy._last_probe.get(str(probe_layer), {})
                probe_z = layer_data.get("z") or None
            episode.records.append(TurnRecord(obs_str, action, log_prob, probe_z))
        else:
            action, _ = opponent_policy.act(
                system_prompt=system_prompt,
                user_prompt=obs_str,
                agent_id=str(player_id),
                steering=None,
            )
        done, _ = env.step(action)
        turn_count += 1
        if done:
            break

    game_rewards, _ = env.close()
    episode.game_rewards = {int(k): float(v) for k, v in game_rewards.items()}
    return episode

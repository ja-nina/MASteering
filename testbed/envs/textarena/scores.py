"""The games' own scores (tokens, chips, points), next to TextArena's win/lose reward.

TextArena's reward only ranks the players: +1 for the best, -1 (or 0) for the rest (PublicGoodsGame:
`winners = [p for p, s in scores.items() if s == max_score]`; Poker, ThreePlayerIPD, SettlersOfCatan: evenly spaced
in [-1, +1] by rank). The payoffs the game itself counts (tokens kept + share of the pot, chips, money, points) are
in its game_state and never reach the reward, so a sum of rewards is not the group's welfare: in public goods more
cooperation means more tokens for everyone but fewer outright winners (main_v2 interim report 2, 2026-11-10).

game_scores(env_id, env) reads that payoff at the end of an episode (or wherever it is stopped): {player_N: number}.
Where it lives, from TextArena's source (envs/<Game>/env.py, 2026-11-10):

  PublicGoodsGame, MarketEntryGame    game_state["total_scores"][pid]       tokens over all rounds
  IteratedPrisonersDilemma,
  ThreePlayerIPD                      game_state["scores"][pid]             payoff points over all rounds
  IteratedUltimatumGame               game_state["player_totals"][pid]      money over all rounds
  SimpleNegotiation                   game_state["inventory_value"][pid]["change"]   value gained by trading
  Poker                               game_state["player_chips"][pid]       chips (constant sum: no welfare)
  SimpleBlindAuction                  auction_results["player_net_worth"]   capital left + value of items won
  HighSociety                         game_state["player_prestige"][pid]    prestige (what decides the winner)
  ThreePlayerGOPS                     game_state["player_scores"][pid]      prize points

No score (the result is only who won): SecretMafia, Debate, Taboo, TruthAndDeception, LiarsDice. SettlersOfCatan's
victory points come from board.get_scores(), keyed by colour; its player -> colour map is not read here (None).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# env id without its version -> (what the score is, the game_state key or a getter(env, game_state))
SCORES: Dict[str, tuple] = {
    "PublicGoodsGame": ("tokens", "total_scores"),
    "MarketEntryGame": ("payoff", "total_scores"),
    "IteratedPrisonersDilemma": ("points", "scores"),
    "ThreePlayerIPD": ("points", "scores"),
    "IteratedUltimatumGame": ("money", "player_totals"),
    "SimpleNegotiation": ("value gained", lambda env, gs: {p: v["change"] for p, v in gs["inventory_value"].items()}),
    "Poker": ("chips", "player_chips"),
    "SimpleBlindAuction": ("net worth", lambda env, gs: _find(env, gs, "auction_results")["player_net_worth"]),
    "HighSociety": ("prestige", "player_prestige"),
    "ThreePlayerGOPS": ("points", "player_scores"),
}


def base_id(env_id: str) -> str:
    return env_id.split("-v")[0]


def score_name(env_id: str) -> Optional[str]:
    spec = SCORES.get(base_id(env_id))
    return spec[0] if spec else None


def _unwrap(env: Any) -> Any:
    """The TextArena env under its wrappers (ta.make wraps; the game's state is on the innermost one)."""
    for _ in range(10):
        if getattr(env, "state", None) is not None and hasattr(env.state, "game_state"):
            return env
        inner = getattr(env, "env", None)
        if inner is None:
            return env
        env = inner
    return env


def _find(env: Any, gs: dict, key: str) -> Any:
    if key in gs:
        return gs[key]
    return getattr(env, key)


def game_scores(env_id: str, env: Any) -> Optional[Dict[str, float]]:
    """{player_N: the game's own score} for the games in SCORES, None for the others or when it cannot be read."""
    spec = SCORES.get(base_id(env_id))
    if spec is None or env is None:
        return None
    env = _unwrap(env)
    gs = getattr(getattr(env, "state", None), "game_state", None) or {}
    get: Callable = spec[1] if callable(spec[1]) else (lambda e, g, k=spec[1]: g[k])
    try:
        raw = get(env, gs)
        return {f"player_{int(p)}": float(v) for p, v in dict(raw).items()}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

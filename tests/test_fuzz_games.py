"""Random play of every symbolic game: legal, illegal and fallback moves, every turn rendered.

Finds crashes on rare paths a single smoke episode never reaches (e.g. Avalon's assassination
after the 5th quest). Each move is drawn from what that game's parser can produce (including out-of-range values)
or is the runner's fallback action 0.
"""
import random

import pytest

from testbed.registry import build_game
from testbed.visual import observe

GAMES = [("gbs", 3, {"num_rounds": 4}), ("gbs", 5, {"hide_group_size": True, "feedback": "directional"}),
         ("beauty_contest", 3, {"num_rounds": 3}), ("harvest", 4, {"num_rounds": 3}),
         ("werewolf", 6, {}), ("werewolf", 8, {}), ("avalon", 5, {}), ("avalon", 6, {}), ("avalon", 8, {}),
         ("hanabi", 3, {}), ("overcooked", 2, {"max_steps": 30}),
         ("tictactoe", 2, {}), ("connect_four", 2, {}), ("gomoku", 2, {})]
EPISODES = 25
MAX_STEPS = 400


def random_action(game, rng, ids, env, obs):
    if rng.random() < 0.15:
        return 0  # the runner's fallback after unparseable answers
    if game in ("gbs", "beauty_contest"):
        return rng.randint(0, 120)  # the parsers reject negative numbers
    if game == "harvest":
        if obs.get("phase") == "discussion":
            return "I will take a few, let's all be careful."
        return rng.randint(0, max(int(obs.get("stock", 0)), 0))  # the parser caps it at the stock
    if game == "werewolf":
        if obs.get("phase") == "day_discussion":
            return {"statement": "I suspect " + rng.choice(ids)}
        return rng.choice(ids)
    if game == "avalon":
        phase = obs.get("phase")
        if phase == "propose":
            return {"team": rng.sample(ids, rng.randint(0, len(ids)))}  # wrong sizes are padded / cut
        if phase == "team_vote":
            return "APPROVE" if rng.random() < 0.8 else "REJECT"  # mostly approve: reach the late quests
        if phase == "quest_vote":
            return rng.choice(["SUCCESS", "FAIL"])
        return rng.choice(ids)  # assassinate
    if game == "hanabi":
        return rng.choice([{"type": "play", "pos": rng.randint(-1, 6)}, {"type": "discard", "pos": rng.randint(-1, 6)},
                           {"type": "hint", "target": rng.choice(ids), "hint_type": "color",
                            "value": rng.choice(["red", "green", "blue", "white", "yellow", "pink"])},
                           {"type": "hint", "target": rng.choice(ids), "hint_type": "rank", "value": rng.randint(0, 6)}])
    if game == "overcooked":
        return rng.choice(["MOVE UP", "MOVE DOWN", "MOVE LEFT", "MOVE RIGHT", "PICK_UP", "DROP", "INTERACT", "STAY"])
    legal = env.legal_actions(None) if hasattr(env, "legal_actions") else None  # board games
    if legal and rng.random() < 0.85:
        return rng.choice(legal)
    return (rng.randint(-1, 20), rng.randint(-1, 20))


@pytest.mark.parametrize("game,players,kwargs", GAMES, ids=[f"{g}_{p}p" for g, p, _ in GAMES])
def test_random_play_never_crashes(game, players, kwargs):
    for ep in range(EPISODES):
        rng = random.Random(f"{game}{players}{ep}")
        env, renderer, _ = build_game(family="symbolic", game_id=game, num_players=players,
                                      env_kwargs={"seed": ep, **kwargs})
        env.reset()
        ids = list(env.agent_ids())
        for step in range(MAX_STEPS):
            pending = env.pending()
            if not pending:
                break
            for aid, obs in pending:
                observe(renderer, "text", obs, aid, env.context)
                if step % 7 == 0:
                    observe(renderer, "image", obs, aid, env.context)
            result = env.submit({aid: random_action(game, rng, ids, env, obs) for aid, obs in pending})
            if result.done:
                break
        if hasattr(env, "roles"):
            env.roles()
        if hasattr(env, "teams"):
            env.teams()

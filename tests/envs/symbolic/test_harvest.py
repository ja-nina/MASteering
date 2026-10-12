from testbed.envs.symbolic.harvest import HarvestAdapter
from testbed.parsers.symbolic.harvest import HarvestParser
from testbed.renderers.symbolic.harvest import HarvestRenderer
from testbed.types import ParsedAction, ParseError


def _talk(env):
    while env.context.extra["phase"] == "discussion":
        (aid, _), = env.pending()
        env.submit({aid: f"hello from {aid}"})


def test_discussion_then_simultaneous_harvest_and_regrowth():
    env = HarvestAdapter(num_players=3, num_rounds=5)
    assert [a for a, _ in env.pending()] == ["player_0"]
    _talk(env)
    assert len(env.pending()) == 3
    res = env.submit({"player_0": 10, "player_1": 10, "player_2": 10})
    assert res.rewards == {"player_0": 10.0, "player_1": 10.0, "player_2": 10.0}
    assert res.info["stock_after"] == 100  # 70 left doubles to 140, capped at capacity
    assert not res.done and env.context.extra["phase"] == "discussion"
    assert env.context.history[0]["messages"][0] == {"speaker": "player_0", "text": "hello from player_0"}


def test_overharvest_is_shared_proportionally_and_collapses():
    env = HarvestAdapter(num_players=2, discussion_turns=0)
    res = env.submit({"player_0": 150, "player_1": 50})
    assert res.info["taken"] == {"player_0": 75, "player_1": 25}
    assert res.info["collapsed"] and res.done and res.info["stock_after"] == 0


def test_game_ends_after_last_round():
    env = HarvestAdapter(num_players=2, num_rounds=2, discussion_turns=0)
    assert not env.submit({"player_0": 5, "player_1": 5}).done
    res = env.submit({"player_0": 5, "player_1": 5})
    assert res.done and res.info["totals"] == {"player_0": 10, "player_1": 10}


def test_renderer_and_parser_follow_phase():
    env = HarvestAdapter(num_players=2)
    obs = env._observation("player_0")
    assert "MESSAGE:" in HarvestRenderer().render(obs, "player_0", env.context)
    parser = HarvestParser()
    assert parser.parse("MESSAGE: take less", obs, "player_0", env.context) == ParsedAction("take less")
    assert isinstance(parser.parse("ok", obs, "player_0", env.context), ParseError)
    _talk(env)
    obs = env._observation("player_0")
    assert "HARVEST:" in HarvestRenderer().render(obs, "player_0", env.context)
    assert "hello from player_1" in HarvestRenderer().render(obs, "player_0", env.context)
    assert parser.parse("HARVEST: 12", obs, "player_0", env.context) == ParsedAction(12)
    assert isinstance(parser.parse("HARVEST: 500", obs, "player_0", env.context), ParseError)

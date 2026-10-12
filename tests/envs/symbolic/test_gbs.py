from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.parsers.symbolic.gbs import GBSParser
from testbed.renderers.symbolic.gbs import GBSRenderer
from testbed.types import ParsedAction, ParseError


def test_signed_error_and_no_reward_when_off():
    env = GBSAdapter(num_players=3, num_rounds=5, target=50)
    env.reset()
    res = env.submit({"player_0": 10, "player_1": 20, "player_2": 30})
    assert res.info == {"group_sum": 60, "error": 10, "direction": "too_high"}
    assert set(res.rewards.values()) == {0.0} and not res.done


def test_exact_sum_rewards_everyone_and_ends():
    env = GBSAdapter(num_players=2, num_rounds=5, target=50)
    env.reset()
    res = env.submit({"player_0": 20, "player_1": 30})
    assert res.done and set(res.rewards.values()) == {1.0}


def test_max_rounds_ends_game():
    env = GBSAdapter(num_players=2, num_rounds=1, target=50)
    env.reset()
    assert env.submit({"player_0": 1, "player_1": 2}).done


def test_target_range_scales_with_players_and_seed():
    a, b = GBSAdapter(num_players=4, seed=1), GBSAdapter(num_players=4, seed=1)
    assert a.target == b.target and 20 <= a.target <= 200


def test_renderer_hides_others_and_picking_variant_hides_group_size():
    env = GBSAdapter(num_players=3, target=50, feedback="exact")
    env.reset()
    env.submit({"player_0": 5, "player_1": 7, "player_2": 9})
    text = GBSRenderer().render(env._observation("player_0"), "player_0", env.context)
    assert "too LOW by 29" in text and "your number=5" in text and "=7" not in text
    pick = GBSAdapter(num_players=3, target=50, hide_group_size=True, feedback="directional")
    pick.reset()
    obs = pick._observation("player_0")
    assert "FINAL GUESS" in GBSRenderer().system_prompt("player_0", obs)
    assert "3 players" not in GBSRenderer().render(obs, "player_0", pick.context)


def test_parser_uses_variant_keyword():
    parser = GBSParser()
    assert parser.parse("NUMBER: 12", {"response_keyword": "NUMBER"}, "p", None) == ParsedAction(12)
    assert parser.parse("FINAL GUESS: 25", {"response_keyword": "FINAL GUESS"}, "p", None) == ParsedAction(25)
    assert isinstance(parser.parse("twelve", {"response_keyword": "NUMBER"}, "p", None), ParseError)

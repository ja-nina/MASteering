from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.renderers.symbolic.gbs import GBSRenderer


def test_beauty_contest_prompt_mentions_rules_and_range():
    env = BeautyContestAdapter(num_players=4, num_rounds=3)
    env.reset()
    obs = env.pending()[0][1]
    r = BeautyContestRenderer()
    sys = r.system_prompt("player_0")
    user = r.render(obs, "player_0", env.context)
    assert "2/3" in sys
    assert "0" in user and "100" in user
    assert "round" in user.lower()


def test_beauty_contest_prompt_includes_history_after_a_round():
    env = BeautyContestAdapter(num_players=2, num_rounds=3)
    env.reset()
    env.submit({"player_0": 10, "player_1": 80})
    obs = env.pending()[0][1]
    user = BeautyContestRenderer().render(obs, "player_0", env.context)
    assert "target" in user.lower()


def test_gbs_prompt_shows_last_direction():
    env = GBSAdapter(num_players=3, num_rounds=9, target=50)
    env.reset()
    env.submit({"player_0": 10, "player_1": 20, "player_2": 30})
    obs = env.pending()[0][1]
    user = GBSRenderer().render(obs, "player_0", env.context)
    assert "higher" in user.lower()

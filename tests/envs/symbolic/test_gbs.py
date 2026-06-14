from testbed.envs.symbolic.gbs import GBSAdapter


def test_feedback_direction_and_reward():
    env = GBSAdapter(num_players=3, num_rounds=5, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 10, "player_1": 20, "player_2": 30})  # median 20 < 50
    assert res.info["median"] == 20
    assert res.info["direction"] == "higher"  # target is higher than median
    assert res.rewards == {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0}
    assert res.done is False


def test_exact_guess_rewarded():
    env = GBSAdapter(num_players=3, num_rounds=5, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 50, "player_1": 10, "player_2": 90})
    assert res.rewards["player_0"] == 1.0
    assert res.rewards["player_1"] == 0.0


def test_group_convergence_ends_game():
    env = GBSAdapter(num_players=3, num_rounds=9, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 50, "player_1": 50, "player_2": 50})  # median 50 == target
    assert res.info["direction"] == "correct"
    assert res.done is True


def test_max_rounds_ends_game():
    env = GBSAdapter(num_players=2, num_rounds=1, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 10, "player_1": 20})
    assert res.done is True

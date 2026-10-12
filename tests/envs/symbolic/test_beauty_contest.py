from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter


def test_pending_returns_all_players_simultaneously():
    env = BeautyContestAdapter(num_players=3, num_rounds=2)
    env.reset()
    pend = env.pending()
    assert sorted(a for a, _ in pend) == ["player_0", "player_1", "player_2"]


def test_target_and_winner():
    env = BeautyContestAdapter(num_players=3, num_rounds=1)
    env.reset()
    # choices 0, 60, 90 -> mean 50 -> target 33.33; closest is 60 (player_1)
    res = env.submit({"player_0": 0, "player_1": 60, "player_2": 90})
    assert res.done is True
    assert res.rewards["player_1"] == 1.0
    assert res.rewards["player_0"] == 0.0
    assert res.rewards["player_2"] == 0.0
    assert round(res.info["target"], 2) == 33.33


def test_tie_splits_reward():
    env = BeautyContestAdapter(num_players=2, num_rounds=1)
    env.reset()
    # both pick 30 -> mean 30 -> target 20; equal distance -> split 0.5 each
    res = env.submit({"player_0": 30, "player_1": 30})
    assert res.rewards["player_0"] == 0.5
    assert res.rewards["player_1"] == 0.5


def test_runs_multiple_rounds_then_done():
    env = BeautyContestAdapter(num_players=2, num_rounds=2)
    env.reset()
    r1 = env.submit({"player_0": 10, "player_1": 20})
    assert r1.done is False
    assert len(env.context.history) == 1
    r2 = env.submit({"player_0": 10, "player_1": 20})
    assert r2.done is True

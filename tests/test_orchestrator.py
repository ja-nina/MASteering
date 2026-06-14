from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.policy.base import StubPolicy
from testbed.steering.noop import NoOpSteering
from testbed.logging_.episode_logger import EpisodeLogger
from testbed.orchestrator import Orchestrator


def test_full_episode_runs_and_logs(tmp_path):
    env = BeautyContestAdapter(num_players=2, num_rounds=2)
    policy = StubPolicy(scripted={
        "player_0": ["CHOICE: 10", "CHOICE: 10"],
        "player_1": ["CHOICE: 20", "CHOICE: 20"],
    })
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="r", episode=0)
    orch = Orchestrator(
        env=env, renderer=BeautyContestRenderer(), parser=BeautyContestParser(),
        policy=policy, steering=NoOpSteering(), logger=logger,
        game="beauty_contest", max_parse_retries=3,
    )
    final = orch.run_episode()
    assert set(final.keys()) == {"player_0", "player_1"}
    # 2 rounds x 2 players = 4 policy calls
    assert len(policy.calls) == 4


def test_orchestrator_reprompts_on_parse_error(tmp_path):
    env = BeautyContestAdapter(num_players=1, num_rounds=1)
    # first completion is junk, second is valid -> one retry
    policy = StubPolicy(scripted={"player_0": ["no number here", "CHOICE: 30"]})
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="r2", episode=0)
    orch = Orchestrator(
        env=env, renderer=BeautyContestRenderer(), parser=BeautyContestParser(),
        policy=policy, steering=NoOpSteering(), logger=logger,
        game="beauty_contest", max_parse_retries=3,
    )
    orch.run_episode()
    assert len(policy.calls) == 2  # initial + 1 retry


def test_orchestrator_falls_back_after_max_retries(tmp_path):
    env = BeautyContestAdapter(num_players=1, num_rounds=1)
    policy = StubPolicy(scripted={}, default="garbage no digits zzz")
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="r3", episode=0)
    orch = Orchestrator(
        env=env, renderer=BeautyContestRenderer(), parser=BeautyContestParser(),
        policy=policy, steering=NoOpSteering(), logger=logger,
        game="beauty_contest", max_parse_retries=2, fallback_action=0,
    )
    final = orch.run_episode()
    # initial + 2 retries = 3 calls, then fallback used
    assert len(policy.calls) == 3
    assert final["player_0"] == 1.0  # only player, wins by default

from testbed.config import build_steering, RunConfig
from testbed.registry import build_game
from testbed.policy.base import StubPolicy
from testbed.steering.noop import NoOpSteering
from testbed.steering.prompt_injection import PromptInjectionSteering
from testbed.steering.activation import ActivationSteering
from testbed.logging_.episode_logger import EpisodeLogger
from testbed.orchestrator import Orchestrator


def test_build_steering_variants():
    assert isinstance(build_steering({"default": "noop", "per_agent": {}}), NoOpSteering)
    pi = build_steering({"default": "prompt_injection",
                         "per_agent": {"player_0": {"system_suffix": " win."}}})
    assert isinstance(pi, PromptInjectionSteering)
    act = build_steering({"default": "activation",
                          "per_agent": {"player_0": {"layer": "l", "vector_path": "v.pt",
                                                     "coefficient": 3.0}}})
    assert isinstance(act, ActivationSteering)


def test_run_config_parses_dict():
    cfg = RunConfig.from_dict({
        "run_id": "t", "game": {"family": "symbolic", "id": "beauty_contest"},
        "episodes": 1, "agents": {"count": 2, "max_parse_retries": 3},
        "model": {"backend": "vllm", "model_id": "m"},
        "steering": {"default": "noop", "per_agent": {}},
        "logging": {"dir": "logs/"},
    })
    assert cfg.game_family == "symbolic"
    assert cfg.game_id == "beauty_contest"
    assert cfg.num_players == 2


def test_end_to_end_symbolic_with_stub(tmp_path):
    env, renderer, parser = build_game(
        family="symbolic", game_id="gbs", num_players=3,
        env_kwargs={"num_rounds": 3, "target": 50})
    policy = StubPolicy(scripted={}, default="GUESS: 50")
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="e2e", episode=0)
    orch = Orchestrator(env=env, renderer=renderer, parser=parser, policy=policy,
                        steering=NoOpSteering(), logger=logger, game="gbs",
                        max_parse_retries=2)
    final = orch.run_episode()
    # all guess 50 == target -> all rewarded, game ends round 1
    assert final == {"player_0": 1.0, "player_1": 1.0, "player_2": 1.0}
    assert (tmp_path / "e2e" / "episode_0.jsonl").exists()

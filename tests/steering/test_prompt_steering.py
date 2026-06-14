from testbed.steering.noop import NoOpSteering
from testbed.steering.prompt_injection import PromptInjectionSteering


def test_noop_is_identity():
    s = NoOpSteering()
    sys, user = s.apply_to_prompt("SYS", "USER", "player_0")
    assert sys == "SYS" and user == "USER"
    assert s.steering_spec("player_0") is None


def test_prompt_injection_appends_per_agent_suffix():
    s = PromptInjectionSteering(per_agent={
        "player_0": {"system_suffix": " Be ruthlessly competitive."},
        "player_1": {"user_prefix": "Remember to cooperate. "},
    })
    sys0, user0 = s.apply_to_prompt("SYS", "USER", "player_0")
    assert sys0.endswith("Be ruthlessly competitive.")
    assert user0 == "USER"
    sys1, user1 = s.apply_to_prompt("SYS", "USER", "player_1")
    assert user1.startswith("Remember to cooperate.")


def test_prompt_injection_unconfigured_agent_unchanged():
    s = PromptInjectionSteering(per_agent={})
    sys, user = s.apply_to_prompt("SYS", "USER", "player_9")
    assert sys == "SYS" and user == "USER"

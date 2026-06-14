from testbed.policy.base import StubPolicy


def test_stub_policy_returns_scripted_completions():
    p = StubPolicy(scripted={"player_0": ["CHOICE: 33", "CHOICE: 22"]})
    assert p.act("s", "u", "player_0", None) == "CHOICE: 33"
    assert p.act("s", "u", "player_0", None) == "CHOICE: 22"


def test_stub_policy_default_when_unscripted():
    p = StubPolicy(scripted={}, default="CHOICE: 0")
    assert p.act("s", "u", "player_5", None) == "CHOICE: 0"


def test_stub_policy_records_calls():
    p = StubPolicy(scripted={}, default="x")
    p.act("SYS", "USER", "player_0", None)
    assert p.calls[-1]["agent_id"] == "player_0"
    assert p.calls[-1]["user"] == "USER"

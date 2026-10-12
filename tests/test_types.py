from testbed.types import (
    StepResult, ParsedAction, ParseError, RenderContext,
)


def test_step_result_fields():
    sr = StepResult(rewards={"player_0": 1.0}, done=False, info={"k": "v"})
    assert sr.rewards["player_0"] == 1.0
    assert sr.done is False
    assert sr.info["k"] == "v"


def test_parse_result_variants():
    ok = ParsedAction(value=42)
    err = ParseError(feedback="bad output, try again")
    assert ok.value == 42
    assert err.feedback == "bad output, try again"


def test_render_context_history():
    ctx = RenderContext(round_index=2, history=[{"target": 27.5}],
                        last_rewards={"player_0": 0.0}, extra={})
    assert ctx.round_index == 2
    assert ctx.history[-1]["target"] == 27.5

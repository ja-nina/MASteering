from testbed.types import (
    StepResult, ParsedAction, ParseError, SteeringSpec, RenderContext,
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


def test_steering_spec_defaults():
    spec = SteeringSpec(method="activation", layer="model.layers.14",
                        vector_path="v.pt", coefficient=8.0)
    assert spec.method == "activation"
    assert spec.coefficient == 8.0
    noop = SteeringSpec(method="noop")
    assert noop.layer is None and noop.vector_path is None and noop.coefficient == 0.0


def test_render_context_history():
    ctx = RenderContext(round_index=2, history=[{"target": 27.5}],
                        last_rewards={"player_0": 0.0}, extra={})
    assert ctx.round_index == 2
    assert ctx.history[-1]["target"] == 27.5

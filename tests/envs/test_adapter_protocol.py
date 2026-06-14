from testbed.envs.adapter import EnvAdapter
from testbed.types import StepResult


class _Toy:
    def reset(self): self._done = False
    def pending(self): return [("player_0", {"obs": 1})]
    def submit(self, actions): return StepResult(rewards={"player_0": 1.0}, done=True)
    def agent_ids(self): return ["player_0"]
    def legal_actions(self, agent_id): return None
    def close(self): return {"player_0": 1.0}


def test_toy_satisfies_protocol():
    a: EnvAdapter = _Toy()
    a.reset()
    pend = a.pending()
    assert pend[0][0] == "player_0"
    res = a.submit({"player_0": 5})
    assert res.done is True
    assert isinstance(a.close(), dict)

import pytest

from testbed.envs.textarena.ta_adapter import TextArenaAdapter
from testbed.renderers.textarena import TextArenaRenderer
from testbed.parsers.textarena import TextArenaParser
from testbed.types import ParsedAction, RenderContext


class FakeTAEnv:
    """Minimal stand-in for a textarena env: 2 turns then done."""
    def __init__(self):
        self._turn = 0
        self.stepped = []
    def reset(self, num_players): self.num_players = num_players; self._turn = 0
    def get_observation(self):
        pid = self._turn % self.num_players
        return pid, f"observation for player {pid}"
    def step(self, action):
        self.stepped.append(action)
        self._turn += 1
        done = self._turn >= 2
        return done, {"info": self._turn}
    def close(self):
        return {0: 1.0, 1: -1.0}, {"game": "fake"}


def test_pending_returns_single_current_player():
    env = TextArenaAdapter(env_id="Fake-v0", num_players=2, _env=FakeTAEnv())
    env.reset()
    pend = env.pending()
    assert len(pend) == 1
    assert pend[0][0] == "player_0"
    assert "player 0" in pend[0][1]


def test_submit_forwards_action_and_advances():
    fake = FakeTAEnv()
    env = TextArenaAdapter(env_id="Fake-v0", num_players=2, _env=fake)
    env.reset()
    env.pending()
    res = env.submit({"player_0": "[Vote] 1"})
    assert fake.stepped == ["[Vote] 1"]
    assert res.done is False
    env.pending()
    res2 = env.submit({"player_1": "ok"})
    assert res2.done is True


def test_close_maps_rewards_to_player_ids():
    env = TextArenaAdapter(env_id="Fake-v0", num_players=2, _env=FakeTAEnv())
    env.reset()
    rewards = env.close()
    assert rewards["player_0"] == 1.0
    assert rewards["player_1"] == -1.0


def test_renderer_is_passthrough():
    r = TextArenaRenderer()
    assert r.render("raw obs string", "player_0", RenderContext()) == "raw obs string"


def test_parser_passes_completion_through():
    p = TextArenaParser()
    res = p.parse("  [Vote] Player 3  ", "obs", "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == "[Vote] Player 3"


@pytest.mark.gpu
def test_real_textarena_smoke():
    ta = pytest.importorskip("textarena")
    env = TextArenaAdapter(env_id="ThreePlayerTicTacToe-v0", num_players=3)
    env.reset()
    pend = env.pending()
    assert len(pend) == 1

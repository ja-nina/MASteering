"""clue_rule (Codenames): the spymasters are told the board-word rule, and a clue TextArena would end the game on is
refused and asked again (at most CLUE_RETRIES times a turn), instead of passed on."""
from types import SimpleNamespace

import pytest

from testbed.envs.textarena.ta_adapter import CLUE_RETRIES, CLUE_RULE, TextArenaAdapter


class _Codenames:
    """Mimics TextArena's Codenames: a board, per-player observations, and the instant loss on a board-word clue."""
    def __init__(self):
        self.board = {"fire": "R", "sticky": "B", "war": "N", "apple": "A"}
        self.steps = []

    def reset(self, num_players, seed=None):
        obs = {p: [(-1, "prompt")] for p in range(num_players)}
        def add_observation(from_id, to_id, message, for_logging=True):
            obs[to_id].append((from_id, message))
        self.state = SimpleNamespace(current_player_id=0, observations=obs, add_observation=add_observation)

    def get_observation(self):
        pid = self.state.current_player_id
        return pid, list(self.state.observations[pid])

    def step(self, action):
        self.steps.append(action)
        word = action.strip("[]").split()[0]
        if any(word in w or w in word for w in self.board):
            return True, {"reason": "subset"}
        self.state.current_player_id = (self.state.current_player_id + 1) % 4
        return False, {}

    def close(self):
        return {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}, {"reason": "subset"}


def _adapter(**kw):
    env = _Codenames()
    ad = TextArenaAdapter("Codenames-v0", 4, _env=env, **kw)
    ad.reset()
    return ad, env


def test_spymasters_are_told_the_rule_operatives_not():
    ad, env = _adapter(clue_rule=True)
    assert [m for _, m in env.state.observations[0]] == ["prompt", CLUE_RULE]
    assert [m for _, m in env.state.observations[2]] == ["prompt", CLUE_RULE]
    assert [m for _, m in env.state.observations[1]] == ["prompt"]


@pytest.mark.parametrize("clue", ["[fire 2]", "[stick 1]", "[warm 1]"])  # exact, part of a word, contains one
def test_board_word_clue_is_refused_and_asked_again(clue):
    ad, env = _adapter(clue_rule=True)
    (aid, _), = ad.pending()
    res = ad.submit({aid: f"{clue}"})
    assert not res.done and env.steps == [] and ad.clue_violations == 1
    (aid2, obs), = ad.pending()
    assert aid2 == aid and "is not allowed" in obs[-1][1]
    res = ad.submit({aid: "[heat 2]"})
    assert not res.done and env.steps == ["[heat 2]"]


def test_after_the_retries_the_clue_goes_to_the_game():
    ad, env = _adapter(clue_rule=True)
    for _ in range(CLUE_RETRIES):
        (aid, _), = ad.pending()
        assert not ad.submit({aid: "[fire 1]"}).done
    (aid, _), = ad.pending()
    res = ad.submit({aid: "[fire 1]"})
    assert res.done and env.steps == ["[fire 1]"] and res.info["clue_violations"] == CLUE_RETRIES


def test_off_by_default_and_operatives_unaffected():
    ad, env = _adapter()
    (aid, _), = ad.pending()
    assert ad.submit({aid: "[fire 1]"}).done
    ad, env = _adapter(clue_rule=True)
    env.state.current_player_id = 1  # an operative guesses a board word: that is the point of its move
    (aid, _), = ad.pending()
    ad.submit({aid: "[fire]"})
    assert env.steps == ["[fire]"] and ad.clue_violations == 0


def test_a_game_without_a_board_fails_at_reset():
    env = _Codenames()
    env.board = None
    with pytest.raises(ValueError, match="no Codenames board"):
        TextArenaAdapter("Poker-v0", 4, _env=env, clue_rule=True).reset()

import pytest

from testbed.registry import build_game
from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.envs.textarena.ta_adapter import TextArenaAdapter


def test_build_beauty_contest():
    env, renderer, parser = build_game(
        family="symbolic", game_id="beauty_contest",
        num_players=3, env_kwargs={"num_rounds": 4})
    assert isinstance(env, BeautyContestAdapter)
    assert isinstance(renderer, BeautyContestRenderer)
    assert isinstance(parser, BeautyContestParser)
    assert env.num_players == 3
    assert env.num_rounds == 4


def test_build_gbs():
    env, renderer, parser = build_game(
        family="symbolic", game_id="gbs", num_players=5, env_kwargs={})
    assert isinstance(env, GBSAdapter)
    assert env.num_players == 5


def test_build_textarena():
    env, renderer, parser = build_game(
        family="textarena", game_id="SecretMafia-v0", num_players=7, env_kwargs={})
    assert isinstance(env, TextArenaAdapter)
    assert env.env_id == "SecretMafia-v0"
    assert env.num_players == 7


def test_unknown_game_raises():
    with pytest.raises(ValueError):
        build_game(family="symbolic", game_id="nope", num_players=2, env_kwargs={})

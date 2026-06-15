"""Verify that the CAA prompt generators produce obs dicts compatible with renderers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.extract_steering_vector import _beauty_contest_prompts, _gbs_prompts
from testbed.registry import build_game


def test_beauty_contest_prompts_render_without_error():
    _, renderer, _ = build_game(
        family="symbolic", game_id="beauty_contest", num_players=4, env_kwargs={})
    pairs = _beauty_contest_prompts(renderer, num_samples=8)
    assert len(pairs) == 8
    for system, user in pairs:
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and "CHOICE:" in user


def test_gbs_prompts_render_without_error():
    _, renderer, _ = build_game(
        family="symbolic", game_id="gbs", num_players=4, env_kwargs={})
    pairs = _gbs_prompts(renderer, num_samples=8)
    assert len(pairs) == 8
    for system, user in pairs:
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and "GUESS:" in user

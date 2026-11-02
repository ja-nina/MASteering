"""TextArenaParser/Renderer with an answer marker: only the text after the last 'ACTION:' reaches the game."""
from testbed.parsers.textarena import TextArenaParser
from testbed.registry import build_game
from testbed.renderers.textarena import TextArenaRenderer
from testbed.types import ParsedAction, ParseError


def parse(text, marker="ACTION"):
    return TextArenaParser(marker).parse(text, None, "player_0", None)


def test_without_marker_the_whole_answer_is_the_action():
    assert TextArenaParser().parse("I hold 3-6. [Call]", None, "player_0", None).value == "I hold 3-6. [Call]"


def test_reasoning_before_the_marker_stays_private():
    assert parse("My hand is 3-6 offsuit, weak.\nACTION: [Fold]").value == "[Fold]"


def test_markdown_and_next_line_and_last_marker():
    assert parse("**ACTION:** [Bet 40]").value == "[Bet 40]"
    assert parse("thinking\nACTION:\n[Call]\nbecause it is cheap").value == "[Call]"
    assert parse("ACTION: [Check]\nwait, no\nACTION: [Raise 20]").value == "[Raise 20]"


def test_bare_bracketed_move_needs_no_marker():
    assert parse("[Call]").value == "[Call]"
    assert parse("[Player 2] [Call]").value == "[Call]"
    assert parse("  [Bid: 3, 4]\n").value == "[Bid: 3, 4]"
    assert isinstance(parse("[Player 1]"), ParseError)          # only a label: no move
    assert isinstance(parse("I hold 7-7. [Call]"), ParseError)  # reasoning outside brackets: marker needed


def test_missing_or_empty_marker_is_retried():
    assert isinstance(parse("no marker here"), ParseError)
    assert isinstance(parse("ACTION:   "), ParseError)
    assert isinstance(parse(""), ParseError)


def test_renderer_tells_the_agent():
    assert "ACTION:" in TextArenaRenderer(marker="ACTION").system_prompt("player_0")
    assert TextArenaRenderer().system_prompt("player_0") == ""


def test_registry_keeps_the_marker_out_of_textarena_kwargs():
    env, renderer, parser = build_game(family="textarena", game_id="Poker-v0", num_players=4,
                                       env_kwargs={"answer_marker": "ACTION"})
    assert parser.marker == "ACTION" and "answer_marker" not in env._env_kwargs

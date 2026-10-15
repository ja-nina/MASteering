"""Keyword answers as the model writes them: markdown, brackets, a final answer after reconsidering. In the
main sweep 20-40% of gbs and image Connect Four turns were rejected for forms like 'FINAL GUESS: [25]'."""
from testbed.parsers.base import extract_int
from testbed.parsers.symbolic.board import BoardParser
from testbed.types import ParsedAction, ParseError


def test_extract_int_forms():
    assert extract_int("FINAL GUESS: [25]", "FINAL GUESS") == 25          # the prompt shows 'FINAL GUESS: [0-50]'
    assert extract_int("**FINAL GUESS: 25**", "FINAL GUESS") == 25
    assert extract_int("**NUMBER**: 12", "NUMBER") == 12
    assert extract_int("NUMBER: <9>", "NUMBER") == 9
    assert extract_int("NUMBER: 3 ... no, NUMBER: 7", "NUMBER") == 7     # the last answer counts
    assert extract_int("NUMBER: -4", "NUMBER") == -4                      # gbs rejects it with feedback
    assert extract_int("I must end with FINAL GUESS: [0-50]", "FINAL GUESS") is None  # the template, not a guess
    assert extract_int("FINAL GUESS: [0-50] ... FINAL GUESS: 30", "FINAL GUESS") == 30
    assert extract_int("my guess is 5", "NUMBER") is None


def test_board_move_forms():
    obs = {"legal": [(5, 0), (5, 3), (4, 4)], "gravity": True}
    move = lambda text: BoardParser().parse(text, obs, "p0", None)
    assert move("**MOVE:** d") == ParsedAction(value=(5, 3))
    assert move("MOVE: A ... no wait, MOVE: D") == ParsedAction(value=(5, 3))
    assert move("Respond with MOVE: <column letter>. MOVE: E") == ParsedAction(value=(4, 4))
    assert isinstance(move("MOVE: <column letter>"), ParseError)
    assert isinstance(move("MOVE: B"), ParseError)  # a full column

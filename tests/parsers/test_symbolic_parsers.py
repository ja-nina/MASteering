from testbed.types import ParsedAction, ParseError, RenderContext
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.parsers.symbolic.gbs import GBSParser


def _obs(low=0, high=100):
    return {"low": low, "high": high}


def test_beauty_parser_extracts_number():
    p = BeautyContestParser()
    res = p.parse("I think 33 is smart. CHOICE: 33", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == 33


def test_beauty_parser_plain_number_fallback():
    p = BeautyContestParser()
    res = p.parse("42", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == 42


def test_beauty_parser_out_of_range_is_error():
    p = BeautyContestParser()
    res = p.parse("CHOICE: 250", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParseError)
    assert "between" in res.feedback.lower()


def test_beauty_parser_no_number_is_error():
    p = BeautyContestParser()
    res = p.parse("I refuse to answer", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParseError)


def test_gbs_parser_extracts_guess():
    p = GBSParser()
    res = p.parse("GUESS: 50", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == 50

"""The condensed report, end to end on synthetic games: plain words and markers, no slopes."""
import random

from scripts import condensed_report, report, summarize_runs
from tests.test_report_dynamics import COEFS, play


def test_condensed_report_in_plain_terms(tmp_path):
    rng = random.Random(0)
    logs = tmp_path / "main"
    for coef in (0,) + COEFS:
        for ep in range(25):
            play(logs, coef, ep, rng)
    out = tmp_path / "agents.csv.gz"
    summarize_runs.main([str(logs), "--out", str(out), "--jobs", "2"])
    page = condensed_report.build(report.load(out), logs=[str(logs)])
    for part in ("id=headline", "id=recipe", "id=synergy", "id=actions", "id=gallery", "id=games"):
        assert part in page
    assert "much more evil" in page and "●●●" in page  # the planted readout effect, in words
    games = page[page.index("id=games"):]
    assert "one agent, vector" in games and " → <b>" in games  # unsteered → steered result
    assert "SD per unit c" not in page and "q=" not in page

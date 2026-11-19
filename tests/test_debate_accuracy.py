"""debate_accuracy.py on a tiny debate_acc sweep played by the fake engine."""
import sys
from pathlib import Path

import yaml

from scripts import run_games
from scripts.make_configs import main as make_configs
from tests.fakes import fake_open_engines
from tests.test_smoke_scripts import _pvf

ROOT = Path(__file__).resolve().parents[1]


def test_debate_accuracy_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PVF_OUT", str(_pvf(tmp_path)))
    sweep = yaml.safe_load((ROOT / "config/sweeps/debate_acc.yaml").read_text(encoding="utf-8"))
    sweep.update(episodes=3, traits=["evil"], coefficients=[1], logging_dir=str(tmp_path / "logs"), models={})
    sweep["conditions"] = {k: sweep["conditions"][k] for k in ("none", "one", "all", "all_random", "prompt_all")}
    for g in sweep["games"].values():
        g["players"] = [1, 3]
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(sweep))
    make_configs([str(tmp_path / "s.yaml"), "--out", str(tmp_path / "runs")])
    assert run_games.main(["--config-dir", str(tmp_path / "runs")], open_engines=fake_open_engines()) == 0
    from scripts import debate_accuracy
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "logs" / "debate_acc"), "--by-n",
                                      "--extra-baseline", str(tmp_path / "logs" / "debate_acc"),
                                      "--out", str(tmp_path / "acc.txt")])
    debate_accuracy.main()
    text = (tmp_path / "acc.txt").read_text(encoding="utf-8")
    for group in ("qa_debate_1p", "qa_debate_3p", "qa_debate_mc_1p", "qa_debate_mc_3p"):
        assert f"\n{group:18}    3" in text  # 3 finished unsteered episodes each
    assert "2. qa_debate, every N pooled" in text and "2. qa_debate_mc, every N pooled" in text
    assert "one evil c=+1" in text and "prompt_all evil c=+1" in text
    assert "2 sample(s):" in text  # the same sweep again as --extra-baseline: two samples per question
    row = next(line for line in text.splitlines() if line.startswith("all evil c=+1"))
    assert len(row.split()) == 13  # 'all' cells carry d rnd and its SE
    row = next(line for line in text.splitlines() if line.startswith("one evil c=+1"))
    assert len(row.split()) == 11  # no random control for 'one'

    from scripts import debate_traces  # the same logs: steered vs unsteered traces of a lone agent
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "logs" / "debate_acc"), "--group", "qa_debate_mc_3p",
                                      "--cell", "all__evil_x1", "--cell", "all_random__evil_x1",
                                      "--out", str(tmp_path / "traces.txt")])
    debate_traces.main()
    text = (tmp_path / "traces.txt").read_text(encoding="utf-8")
    assert "1. qa_debate_mc_3p, 3 questions every run finished" in text
    for run in ("none", "all__evil_x1", "all_random__evil_x1"):
        assert any(line.split()[:1] == [run] and len(line.split()) == 9 for line in text.splitlines()), run
    assert "2. Questions unsteered got right and all__evil_x1 got wrong:" in text


def test_by_question_counts_each_question_once():
    from scripts.debate_accuracy import by_question
    # question 0 met at three N (all +1), question 1 once (-1): per question +1 and -1, so mean 0, not +0.5
    m, s = by_question([(0, 1.0), (0, 1.0), (0, 1.0), (1, -1.0)])
    assert m == 0.0 and abs(s - 1.0) < 1e-12  # SE over 2 questions: sd sqrt(2) / sqrt(2)

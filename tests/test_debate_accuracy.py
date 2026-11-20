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
    sec3 = text.split("3. qa_debate_mc: does")[1].split("\n4. ")[0]
    row = next(line for line in sec3.splitlines() if line.startswith("all evil c=+1"))
    assert len(row.split()) == 10 and row.split()[3] == "3"  # 3 questions met at N = 1 and N = 3
    sec4 = text.split("4. qa_debate_mc, N >= 2 pooled")[1].split("\n3. ")[0]
    assert any(line.split()[:2] == ["none", "3"] for line in sec4.splitlines())  # the 3 unsteered 3p episodes
    assert any(line.startswith("all_random evil c=+1") for line in sec4.splitlines())

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


def test_switches_conformity_and_unanimity(tmp_path):
    import json
    from scripts.debate_accuracy import episode, slope
    rounds = [{"p0": "A", "p1": "B", "p2": "B"},   # p0 alone on A
              {"p0": "B", "p1": "B", "p2": "C"},   # p0 -> B (others' majority: conforms), p2 -> C (not)
              {"p0": "C", "p1": "C", "p2": "C"}]   # p0 -> C (others B/C tied: counts), p1 -> C (others B/C tied)
    path = tmp_path / "episode_0.jsonl"
    with open(path, "w") as f:
        for t, ans in enumerate(rounds):
            info = {"answers": ans, "correct": {p: a == "A" for p, a in ans.items()},
                    "majority_correct": float(sum(a == "A" for a in ans.values()) > 1)}
            for p in ans:
                f.write(json.dumps({"turn": t, "agent_id": p, "info": info}) + "\n")
    (tmp_path / "episode_0.summary.json").write_text("{}")
    e = episode(path)
    assert abs(e["switch"] - 4 / 6) < 1e-12 and abs(e["to_group"] - 3 / 6) < 1e-12
    assert e["unan"] == 1.0 and e["unan_wrong"] == 1.0  # all on C, and A is right
    assert abs(e["rw"] - 1 / 3) < 1e-12 and e["wr"] == 0.0  # p0 was right in round 1, wrong at the end
    assert slope([(1, 0.0), (3, -0.2), (5, -0.4)]) == -0.1


def test_by_question_counts_each_question_once():
    from scripts.debate_accuracy import by_question
    # question 0 met at three N (all +1), question 1 once (-1): per question +1 and -1, so mean 0, not +0.5
    m, s = by_question([(0, 1.0), (0, 1.0), (0, 1.0), (1, -1.0)])
    assert m == 0.0 and abs(s - 1.0) < 1e-12  # SE over 2 questions: sd sqrt(2) / sqrt(2)


def test_late_steering_starts_in_round_two(tmp_path, monkeypatch):
    """debate_late's 'late' condition: no vector and no persona in round 1, the trait's vector from round 2 on;
    the analysis pairs late with late_random."""
    import json
    monkeypatch.setenv("PVF_OUT", str(_pvf(tmp_path)))
    sweep = yaml.safe_load((ROOT / "config/sweeps/debate_late.yaml").read_text(encoding="utf-8"))
    sweep.update(episodes=2, traits=["evil"], coefficients=[1], logging_dir=str(tmp_path / "logs"), models={})
    sweep["games"]["qa_debate_hard"].update(players=[1, 3], env_kwargs={"dataset": "mmlu_pro", "num_rounds": 3})
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(sweep))
    make_configs([str(tmp_path / "s.yaml"), "--out", str(tmp_path / "runs")])
    assert run_games.main(["--config-dir", str(tmp_path / "runs")], open_engines=fake_open_engines()) == 0
    run = tmp_path / "logs" / "debate_acc" / "qa_debate_hard_3p" / "qa_debate_hard_3p__late__evil_x1"
    recs = [json.loads(line) for f in sorted(run.glob("episode_*.jsonl")) for line in open(f, encoding="utf-8")]
    assert {r["turn"] for r in recs} == {0, 1, 2}
    assert all(r["steering"] is None for r in recs if r["turn"] == 0)
    later = [r["steering"] for r in recs if r["turn"] > 0]
    assert later and all(s and s["from_turn"] == 1 and s["coeff"] == 1 and s["layer"] is not None for s in later)
    full = tmp_path / "logs" / "debate_acc" / "qa_debate_hard_3p" / "qa_debate_hard_3p__all__evil_x1"
    steers = [json.loads(line)["steering"] for f in full.glob("episode_*.jsonl") for line in open(f, encoding="utf-8")]
    assert steers and all(s and "from_turn" not in s for s in steers)  # 'all': every turn, logged as before

    from scripts import debate_accuracy
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "logs" / "debate_acc"), "--out", str(tmp_path / "acc.txt")])
    debate_accuracy.main()
    text = (tmp_path / "acc.txt").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("late evil c=+1"))
    assert len(row.split()) == 13  # late carries d rnd against late_random, as 'all' does against all_random
    sec3 = text.split("\n3. ")[1].split("\n4. ")[0]
    assert any(line.startswith("late evil c=+1") for line in sec3.splitlines())


def test_fixed_layer_sweep_steers_at_that_layer(tmp_path, monkeypatch):
    """debate_l12: every trait at layer 12 (here 2, the fake vectors have 3), not its auto layer (evil's is 1);
    the analysis pairs l12 with l12_random."""
    import json
    monkeypatch.setenv("PVF_OUT", str(_pvf(tmp_path)))
    sweep = yaml.safe_load((ROOT / "config/sweeps/debate_l12.yaml").read_text(encoding="utf-8"))
    assert sweep["coefficients"] == [-0.5, 0.5]
    assert {k: (v or {}).get("layer") for k, v in sweep["conditions"].items()} == {"none": None, "l12": 12,
                                                                                    "l12_random": 12}
    for v in sweep["conditions"].values():
        if v:
            v["layer"] = 2
    sweep.update(episodes=2, traits=["evil"], logging_dir=str(tmp_path / "logs"), models={})
    sweep["games"]["qa_debate_hard"].update(players=[1, 3], env_kwargs={"dataset": "mmlu_pro", "num_rounds": 3})
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(sweep))
    make_configs([str(tmp_path / "s.yaml"), "--out", str(tmp_path / "runs")])
    assert run_games.main(["--config-dir", str(tmp_path / "runs")], open_engines=fake_open_engines()) == 0
    for cond in ("l12", "l12_random"):
        run = tmp_path / "logs" / "debate_acc" / "qa_debate_hard_3p" / f"qa_debate_hard_3p__{cond}__evil_x0.5"
        steers = [json.loads(line)["steering"] for f in run.glob("episode_*.jsonl") for line in open(f, encoding="utf-8")]
        assert steers and all(s and s["layer"] == 2 and s["coeff"] == 0.5 for s in steers), cond

    from scripts import debate_accuracy
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "logs" / "debate_acc"), "--out", str(tmp_path / "acc.txt")])
    debate_accuracy.main()
    text = (tmp_path / "acc.txt").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("l12 evil c=+0.5"))
    assert len(row.split()) == 13  # d rnd against l12_random

import json

import pytest
import torch
import yaml

from testbed.config import RunConfig, build_jobs
from testbed.engine import Monitor
from testbed.runner import run_jobs
from testbed.steering import VectorStore
from tests.fakes import HIDDEN, FakeGenerator, FakeReader, fake_open_engines, make_vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(make_vectors(tmp_path / "vec"), cache_dir=tmp_path / "cache")


def _cfg(tmp_path, run_id, game, players, steering=None, monitor=None, **kw):
    family = kw.pop("family", "symbolic")
    return RunConfig.from_dict({"run_id": run_id, "model": "qwen3.5-9b", "vectors_dir": str(tmp_path / "vec"),
                                "game": {"family": family, "id": game, "players": players,
                                         "env_kwargs": kw.pop("env_kwargs", {})},
                                "episodes": kw.pop("episodes", 2), "steering": steering, "monitor": monitor,
                                "logging_dir": str(tmp_path / "logs"), **kw})


def _records(cfg, ep=0):
    path = cfg.run_dir / f"episode_{ep}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_monitor_projects_mean_residual_on_unit_vectors(store):
    monitor = Monitor(store, layers=[1, 2], traits=["evil", "empathy"])
    rows = {1: torch.zeros(3, HIDDEN), 2: torch.zeros(3, HIDDEN)}
    rows[1][:, 0] = torch.tensor([1.0, 2.0, 3.0])  # mean 2 along evil's main axis
    scores = monitor.scores(rows)
    unit = store.vectors("evil")[1] / store.vectors("evil")[1].norm()
    assert scores["1"]["evil"] == pytest.approx(2 * unit[0].item(), abs=1e-4)
    assert scores["2"] == {"evil": 0.0, "empathy": 0.0}
    assert monitor.scores({1: torch.zeros(0, HIDDEN), 2: torch.zeros(0, HIDDEN)}) == {}


def test_concurrent_runs_share_generate_calls(tmp_path, store):
    configs = [_cfg(tmp_path, "gbs_none", "gbs", 3),
               _cfg(tmp_path, "gbs_all", "gbs", 3, steering={"trait": "evil", "coefficient": 2.5})]
    jobs = build_jobs(configs, store)
    gen = FakeGenerator()
    counts = run_jobs(jobs, gen, concurrent=8)
    assert counts == {"ok": 4, "failed": 0}
    # GBS is simultaneous: the first step holds every player of all four episodes
    assert len(gen.batches[0]) == 4 * 3
    steered = [r for b in gen.batches for r in b if r.steer is not None]
    assert steered and all(r.key[0] == 1 for r in steered) and all(r.steer.layer == 1 for r in steered)
    rec = _records(configs[1])[0]
    assert rec["steering"]["label"] == "evil@L1x2.5" and rec["steering"]["coeff"] == 2.5
    assert _records(configs[0])[0]["steering"] is None
    summary = json.loads((configs[1].run_dir / "episode_0.summary.json").read_text())
    assert set(summary["steered"]) == {"player_0", "player_1", "player_2"}
    run = json.loads((configs[1].run_dir / "run.json").read_text())
    assert run["steering_resolved"][0]["layer"] == 1


def test_paired_conditions_play_the_same_game(tmp_path, store):
    a, b = (_cfg(tmp_path, rid, "werewolf", 6, episodes=1) for rid in ("ww_a", "ww_b"))
    run_jobs(build_jobs([a, b], store), FakeGenerator())
    roles = [json.loads((c.run_dir / "episode_0.summary.json").read_text())["player_roles"] for c in (a, b)]
    assert roles[0] == roles[1] and roles[0]


def test_team_steering_in_werewolf(tmp_path, store):
    cfg = _cfg(tmp_path, "ww_evil", "werewolf", 6, episodes=1, max_parse_retries=8,
               steering={"team": "evil", "trait": "evil", "coefficient": 2.5, "layer": 2})
    gen = FakeGenerator()
    assert run_jobs(build_jobs([cfg], store), gen) == {"ok": 1, "failed": 0}
    summary = json.loads((cfg.run_dir / "episode_0.summary.json").read_text())
    wolves = {a for a, t in summary["teams"].items() if t == "evil"}
    assert set(summary["steered"]) == wolves
    for r in (r for b in gen.batches for r in b):
        assert (r.steer is not None) == (r.key[3] in wolves)


def test_parse_retry_appends_feedback_then_falls_back(tmp_path, store):
    cfg = _cfg(tmp_path, "bc", "beauty_contest", 2, episodes=1, max_parse_retries=2,
               env_kwargs={"num_rounds": 1})
    gen = FakeGenerator(respond=lambda r: "no idea" if r.key[3] == "player_0" else "CHOICE: 40")
    run_jobs(build_jobs([cfg], store), gen)
    p0 = [r for b in gen.batches for r in b if r.key[3] == "player_0"]
    assert [r.key[4] for r in p0] == [0, 1, 2]
    assert "CHOICE: <number>" in p0[1].user and p0[0].seed != p0[1].seed
    rec = next(r for r in _records(cfg) if r["agent_id"] == "player_0")
    assert rec["parse_ok"] is False and rec["parse_retries"] == 2 and rec["parsed_action"] == 0
    # every attempt is logged with its prompt, output and parse error
    assert [a["attempt"] for a in rec["attempts"]] == [0, 1, 2]
    assert all(a["completion"] == "no idea" and a["parse_error"] for a in rec["attempts"])
    assert "CHOICE: <number>" not in rec["attempts"][0]["user_prompt"]
    assert "CHOICE: <number>" in rec["attempts"][1]["user_prompt"] and rec["user_prompt"] == rec["attempts"][2]["user_prompt"]
    assert rec["model_input"].startswith("<system>") and rec["user_prompt"] in rec["model_input"]
    ok = next(r for r in _records(cfg) if r["agent_id"] == "player_1")
    assert len(ok["attempts"]) == 1 and ok["attempts"][0]["parse_error"] is None
    trace = (cfg.run_dir / "episode_0.trace.txt").read_text(encoding="utf-8")
    assert trace.count("REJECTED") == 2


def test_monitor_scores_are_logged_per_turn(tmp_path, store):
    cfg = _cfg(tmp_path, "harvest_mon", "harvest", 3, episodes=1, monitor={"layers": "auto"},
               env_kwargs={"num_rounds": 2})
    layers = [1, 2, 3]
    reader = FakeReader(layers)
    run_jobs(build_jobs([cfg], store), FakeGenerator(), reader, Monitor(store, layers))
    recs = _records(cfg)
    assert reader.calls == len({r["turn"] for r in recs})
    assert set(recs[0]["monitor"]) == {"1", "2", "3"}
    assert set(recs[0]["monitor"]["1"]) == {"evil", "sarcastic", "empathy"}


def test_failing_episode_does_not_stop_the_others(tmp_path, store):
    good = _cfg(tmp_path, "gbs_ok", "gbs", 2, episodes=1)
    bad = _cfg(tmp_path, "gbs_bad", "gbs", 2, episodes=1, steering={"team": "evil", "trait": "evil", "coefficient": 1})
    assert run_jobs(build_jobs([good, bad], store), FakeGenerator()) == {"ok": 1, "failed": 1}


def test_run_games_cli_resumes_and_shards(tmp_path, monkeypatch):
    from scripts import run_games
    make_vectors(tmp_path / "vec")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for name, steering in (("none", None), ("one", {"select": "one", "trait": "sarcastic", "coefficient": 2.5})):
        (cfg_dir / f"{name}.yaml").write_text(yaml.safe_dump({
            "run_id": f"gbs_{name}", "model": "qwen3.5-9b", "vectors_dir": str(tmp_path / "vec"),
            "game": {"family": "symbolic", "id": "gbs", "players": 3}, "episodes": 3, "steering": steering,
            "monitor": {"layers": [2]}, "logging_dir": str(tmp_path / "logs")}))
    gen = FakeGenerator()
    args = ["--config-dir", str(cfg_dir), "--num-shards", "2"]
    assert run_games.main(args + ["--shard", "0"], open_engines=fake_open_engines(gen)) == 0
    done = sorted(p.parent.name for p in (tmp_path / "logs").rglob("*.summary.json"))
    # every other entry of the episode-major order (none 0, one 0, none 1, ...): none's episodes 0, 1, 2
    assert done == ["gbs_none"] * 3
    assert run_games.main(args + ["--shard", "1"], open_engines=fake_open_engines(gen)) == 0
    calls = len(gen.batches)
    assert run_games.main(["--config-dir", str(cfg_dir)], open_engines=fake_open_engines(gen)) == 0
    assert len(gen.batches) == calls  # everything done: resume skips it all
    assert len(list((tmp_path / "logs").rglob("*.summary.json"))) == 6


def test_summarize_runs_csv(tmp_path, store):
    import csv
    from scripts.summarize_runs import main as summarize
    cfg = _cfg(tmp_path, "ww_evil", "werewolf", 6, episodes=1, max_parse_retries=8, monitor={"layers": [1, 2, 3]},
               steering={"team": "evil", "trait": "evil", "coefficient": 2.5})
    run_jobs(build_jobs([cfg], store), FakeGenerator(), FakeReader([1, 2, 3]), Monitor(store, [1, 2, 3]))
    out = tmp_path / "agents.csv"
    summarize([str(tmp_path / "logs"), "--out", str(out)])
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 6
    assert all((r["steered"] == "True") == (r["team"] == "evil") for r in rows)
    assert rows[0]["condition_team"] == "evil" and "mon_evil" in rows[0]


def test_summary_total_rewards_sum_every_step(tmp_path, store):
    """Games that pay every round (harvest, beauty contest): the summary's total_rewards is the episode's score,
    final_rewards only the last step's."""
    configs = [_cfg(tmp_path, "hv", "harvest", 3, episodes=1), _cfg(tmp_path, "bc", "beauty_contest", 3, episodes=1)]
    assert run_jobs(build_jobs(configs, store), FakeGenerator()) == {"ok": 2, "failed": 0}
    hv, bc = (json.loads((c.run_dir / "episode_0.summary.json").read_text()) for c in configs)
    assert hv["total_rewards"] == {a: float(t) for a, t in hv["final_info"]["totals"].items()}
    summed = {}
    for rec in _records(configs[1]):
        summed[rec["agent_id"]] = summed.get(rec["agent_id"], 0.0) + rec["reward"]
    assert bc["total_rewards"] == summed and bc["turns"] == 5

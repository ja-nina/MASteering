"""summarize_runs reads the episode summaries' agent_stats; older episodes (no agent_stats) get them from
their JSONL once, written back into the summary, and give the same rows."""
import csv
import gzip
import json

from scripts import summarize_runs
from testbed.logging_.episode_logger import EpisodeLogger


def play(logs, run_id, ep):
    """One logged episode: 3 turns of p0 and p1, monitored at layers 8 and 16."""
    d = logs / run_id.split("__")[0]
    d.mkdir(parents=True, exist_ok=True)
    (d / run_id).mkdir(exist_ok=True)
    (d / run_id / "run.json").write_text(json.dumps({"run_id": run_id, "game": {"id": "g", "players": 2},
                                                     "steering": {"select": "one", "trait": "evil", "coefficient": 1}}))
    log = EpisodeLogger(str(d), run_id, ep)
    for turn in range(3):
        for aid in ("p0", "p1"):
            log.log_step(game="g", turn=turn, agent_id=aid, system_prompt="s", user_prompt="u",
                         completion="x" * (turn + 1), parse_retries=0, reward=0.0,
                         parsed_action=f"I bid. [Bid {10 * (turn + 1)}]" if aid == "p0" else ["APPROVE", "REJECT", 3][turn],
                         parse_ok=turn != 2 or aid == "p0", monitor={"8": {"evil": 9.0}, "16": {"evil": float(turn)}})
    log.close(summary={"final_rewards": {"p0": 1, "p1": 0}})
    return d / run_id / f"episode_{ep}.summary.json"


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [{k: v for k, v in r.items() if k != "episode"} for r in csv.DictReader(f)]


def test_summary_stats_match_jsonl_and_cache(tmp_path, capsys):
    logs, out = tmp_path / "main", str(tmp_path / "res" / "agents.csv.gz")
    new = play(logs, "g_2p__none", 0)
    summary = json.loads(new.read_text())
    p0, p1 = summary["agent_stats"]["p0"], summary["agent_stats"]["p1"]
    assert p1["monitor"]["16"]["evil"] == 1.0
    assert p1["phase_layer"] == "16" and [ph["evil"] for ph in p1["phases"]] == [0.0, 1.0, 2.0]  # turn i in third i
    assert p0["actions"] == {"n": 3, "numeric": 20.0, "labels": {"bid #": 3}}
    assert p1["actions"]["labels"] == {"approve": 1, "reject": 1}  # turn 2 of p1: parse failed, not counted
    old = play(logs, "g_2p__none", 1)  # logged before summaries had agent_stats
    old.write_text(json.dumps({k: v for k, v in summary.items() if k != "agent_stats"}))
    summarize_runs.main([str(logs), "--out", out, "--jobs", "2"])
    rows = read(out)
    assert rows[:2] == rows[2:]  # episode 0 (from its summary) == episode 1 (from its JSONL)
    r = rows[1]
    assert (r["agent"], r["turns"], r["parse_fallbacks"], r["answer_chars"], r["monitor_layer"], r["mon_evil"]) == \
        ("p1", "3", "1", "2.0", "16", "1.0")
    assert (r["mon_early_evil"], r["mon_late_evil"], r["act_n"], json.loads(r["act_labels"])) ==         ("0.0", "2.0", "2", {"approve": 1, "reject": 1})
    assert "(1 older ones read from their JSONL" in capsys.readouterr().out
    assert json.loads(old.read_text())["agent_stats"] == summary["agent_stats"]  # written back

    summarize_runs.main([str(logs), "--out", out, "--jobs", "2"])  # rerun: summaries only
    assert "(0 older ones read from their JSONL" in capsys.readouterr().out
    assert read(out) == rows


def test_large_output_split_into_parts(tmp_path, monkeypatch):
    from scripts import report
    out = str(tmp_path / "agents.csv.gz")
    rows = [{"a": i, "b": "x"} for i in range(12000)]
    monkeypatch.setattr(summarize_runs, "MAX_PART_BYTES", 1)  # a new part at every check (each 5000 rows)
    assert summarize_runs.write_csv(out, ["a", "b"], rows) == 3
    assert [p.name for p in summarize_runs.csv_parts(out)] == ["agents.csv.gz", "agents-2.csv.gz", "agents-3.csv.gz"]
    assert [int(r["a"]) for r in report.read_csv(out)] == list(range(12000))
    monkeypatch.setattr(summarize_runs, "MAX_PART_BYTES", 10 ** 9)
    assert summarize_runs.write_csv(out, ["a", "b"], rows) == 1  # stale parts of the earlier output removed
    assert summarize_runs.csv_parts(out) == [tmp_path / "agents.csv.gz"]

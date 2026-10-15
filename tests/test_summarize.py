"""summarize_runs reads the episode summaries' agent_stats; older episodes (no agent_stats) come from their
JSONL, cached, and give the same rows."""
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
    (d / run_id / "run.json").write_text(json.dumps({"run_id": run_id, "game": {"id": "g", "players": 2}}))
    log = EpisodeLogger(str(d), run_id, ep)
    for turn in range(3):
        for aid in ("p0", "p1"):
            log.log_step(game="g", turn=turn, agent_id=aid, system_prompt="s", user_prompt="u",
                         completion="x" * (turn + 1), parsed_action=None, parse_retries=0, reward=0.0,
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
    assert summary["agent_stats"]["p1"]["monitor"]["16"]["evil"] == 1.0
    old = play(logs, "g_2p__none", 1)  # logged before summaries had agent_stats
    old.write_text(json.dumps({k: v for k, v in summary.items() if k != "agent_stats"}))
    summarize_runs.main([str(logs), "--out", out, "--jobs", "2"])
    rows = read(out)
    assert rows[:2] == rows[2:]  # episode 0 (from its summary) == episode 1 (from its JSONL)
    r = rows[1]
    assert (r["agent"], r["turns"], r["parse_fallbacks"], r["answer_chars"], r["monitor_layer"], r["mon_evil"]) == \
        ("p1", "3", "1", "2.0", "16", "1.0")
    assert "(1 older ones read from their JSONL)" in capsys.readouterr().out

    summarize_runs.main([str(logs), "--out", out, "--jobs", "2"])  # rerun: the old episode comes from the cache
    assert "(0 older ones read from their JSONL)" in capsys.readouterr().out
    assert read(out) == rows

    cache = tmp_path / "res" / "agents.csv.gz.cache.pkl"
    cache.write_bytes(cache.read_bytes()[:20])  # cut off (Ctrl+C mid-save): read again, not a crash
    summarize_runs.main([str(logs), "--out", out, "--jobs", "2"])
    assert "unreadable" in capsys.readouterr().out and read(out) == rows

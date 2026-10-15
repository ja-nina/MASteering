"""summarize_runs: episode logs read in parallel and cached; a rerun reads only new or grown logs."""
import csv
import gzip
import json

from scripts import summarize_runs


def write_run(root, run_id, episodes, steering=None):
    d = root / run_id.split("__")[0] / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "game": {"id": "g", "players": 2}, "steering": steering}))
    for ep in episodes:
        with open(d / f"episode_{ep}.jsonl", "w") as f:
            for turn in range(3):
                for aid in ("p0", "p1"):
                    f.write(json.dumps({"agent_id": aid, "parse_ok": turn != 2 or aid == "p0", "completion": "x" * (turn + 1),
                                        "monitor": {"8": {"evil": 9.0}, "16": {"evil": float(turn)}}}) + "\n")
        (d / f"episode_{ep}.summary.json").write_text(json.dumps({"final_rewards": {"p0": 1, "p1": 0}}))
    (d / "episode_9.jsonl").write_text("")  # unfinished episode: no summary, not a row


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_rows_parallel_and_cached(tmp_path, capsys, monkeypatch):
    logs, out = tmp_path / "main", str(tmp_path / "res" / "agents.csv.gz")
    write_run(logs, "g_2p__none", (0, 1))
    write_run(logs, "g_2p__one__evil_x1", (0,), {"select": "one", "trait": "evil", "coefficient": 1.0})
    summarize_runs.main([str(logs), "--out", out, "--jobs", "2"])
    rows = read(out)
    assert len(rows) == 6
    r = next(r for r in rows if r["run_id"] == "g_2p__none" and r["agent"] == "p1")
    assert (r["turns"], r["parse_fallbacks"], r["answer_chars"], r["monitor_layer"], r["mon_evil"]) == \
        ("3", "1", "2.0", "16", "1.0")
    assert "3 episode logs to read (0 cached" in capsys.readouterr().out

    write_run(logs, "g_2p__none", (2,))  # one new episode: only it is read
    monkeypatch.setattr(summarize_runs, "episode_stats", lambda *a: (_ for _ in ()).throw(AssertionError))
    summarize_runs.main([str(logs), "--out", out, "--jobs", "1"])

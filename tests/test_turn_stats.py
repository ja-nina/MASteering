import csv
import gzip
import json


def write_episode(d, recs):
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_id": "g_2p__one__evil_x1"}))
    (d / "episode_0.summary.json").write_text("{}")
    (d / "episode_0.jsonl").write_text("\n".join(json.dumps(r) for r in recs))


def test_turn_stats_breaks_bins_and_echo(tmp_path):
    from scripts.turn_stats import main
    a_text = "the lake is almost empty so we take less"
    recs = [
        {"turn": 0, "agent_id": "player_0", "completion": a_text, "parse_ok": True, "finish_reason": "stop",
         "attempts": [{"n_tokens": 100, "finish_reason": "stop"}]},
        # simultaneous turn: player_1 has not seen player_0's text yet
        {"turn": 0, "agent_id": "player_1", "completion": a_text, "parse_ok": True, "finish_reason": "stop",
         "attempts": [{"n_tokens": 100, "finish_reason": "stop"}]},
        # next turn player_1 repeats player_0 (echo 1.0); unparseable, every attempt cut off
        {"turn": 1, "agent_id": "player_1", "completion": a_text, "parse_ok": "False", "finish_reason": "length",
         "attempts": [{"n_tokens": 1024, "finish_reason": "length"}, {"n_tokens": 1024, "finish_reason": "length"}]},
        {"turn": 1, "agent_id": "player_0", "completion": "completely different words here today friends",
         "parse_ok": False, "finish_reason": "stop", "attempts": [{"n_tokens": 300, "finish_reason": "stop"}]},
    ]
    write_episode(tmp_path / "logs" / "g_2p" / "g_2p__one__evil_x1", recs)
    out = tmp_path / "ts.csv.gz"
    main([str(tmp_path / "logs"), "--out", str(out), "--jobs", "1"])
    rows = {r["agent"]: r for r in csv.DictReader(gzip.open(out, "rt"))}
    p0, p1 = rows["player_0"], rows["player_1"]
    assert p1["run_id"] == "g_2p__one__evil_x1" and p1["episode"] == "0"
    assert (p1["turns"], p1["failed"], p1["cut_turns"], p1["failed_cut"]) == ("2", "1", "1", "1")
    assert (p0["failed"], p0["failed_cut"]) == ("1", "0")          # format error, not cut off
    assert p1["t_b0"] == "1" and p1["t_b5"] == "1" and p1["f_b5"] == "1"
    assert p0["f_b2"] == "1"                                       # 300 tokens: bin (256, 512]
    assert json.loads(p1["echo"])["player_0"] == 0.5               # turn 0: 0, turn 1: 1
    assert json.loads(p0["echo"])["player_1"] == 0.0

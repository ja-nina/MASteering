"""Episodes logged before summaries had total_rewards: backfill_rewards writes their score into a side file
(episode_N.rewards.json; the summary is left as it is) for the games that pay every step, and summarize_runs
takes the reward from total_rewards, then the side file, then final_rewards (the last step's)."""
import csv
import gzip
import json

from scripts import backfill_rewards, summarize_runs
from testbed.logging_.episode_logger import EpisodeLogger


def old_episode(logs, game, steps, summary):
    """An episode of `game` logged the old way: no total_rewards in its summary. `steps`: per step {agent: reward}."""
    run_id = f"{game}_2p__none"
    d = logs / f"{game}_2p"
    (d / run_id).mkdir(parents=True)
    (d / run_id / "run.json").write_text(json.dumps({"run_id": run_id, "game": {"id": game, "players": 2}}))
    log = EpisodeLogger(str(d), run_id, 0)
    for turn, rewards in enumerate(steps):
        for aid, r in rewards.items():
            log.log_step(game=game, turn=turn, agent_id=aid, system_prompt="s", user_prompt="u", completion="c",
                         parsed_action=1, parse_retries=0, reward=r)
    log.close(summary={"final_rewards": steps[-1], **summary})
    return d / run_id / "episode_0.summary.json"


def rows(out):
    with gzip.open(out, "rt", encoding="utf-8") as f:
        return {(r["game"], r["agent"]): (r["reward"], r["reward_from"]) for r in csv.DictReader(f)}


def test_backfill_then_summarize(tmp_path, capsys):
    logs, out = tmp_path / "main", str(tmp_path / "agents.csv.gz")
    two = [{"p0": 3.0, "p1": 1.0}, {"p0": 0.0, "p1": 2.0}]
    hv = old_episode(logs, "harvest", two, {"final_info": {"totals": {"p0": 3, "p1": 3}}})
    old_episode(logs, "beauty_contest", two, {"final_info": {}})
    old_episode(logs, "overcooked", [{"p0": 0.0, "p1": 0.0}], {"final_info": {"score": 2}})
    old_episode(logs, "avalon", [{"p0": 0.0, "p1": 0.0}, {"p0": 1.0, "p1": 0.0}], {"final_info": {}})
    raw = hv.read_text()

    summarize_runs.main([str(logs), "--out", out, "--jobs", "1"])
    assert "WARNING: 6 rows of games that pay every step" in capsys.readouterr().out
    assert rows(out)[("harvest", "p0")] == ("0.0", "last_step")

    backfill_rewards.main([str(logs), "--jobs", "1"])
    assert hv.read_text() == raw  # the summary itself is not touched
    assert json.loads(hv.with_name("episode_0.rewards.json").read_text()) == \
        {"total_rewards": {"p0": 3.0, "p1": 3.0}, "source": "final_info.totals"}
    summarize_runs.main([str(logs), "--out", out, "--jobs", "1"])
    assert "WARNING" not in capsys.readouterr().out
    assert rows(out) == {
        ("harvest", "p0"): ("3.0", "backfill"), ("harvest", "p1"): ("3.0", "backfill"),
        ("beauty_contest", "p0"): ("3.0", "backfill"), ("beauty_contest", "p1"): ("3.0", "backfill"),
        ("overcooked", "p0"): ("2.0", "backfill"), ("overcooked", "p1"): ("2.0", "backfill"),
        ("avalon", "p0"): ("1.0", "last_step"), ("avalon", "p1"): ("0.0", "last_step"),  # pays once: unchanged
    }

    backfill_rewards.main([str(logs), "--jobs", "1"])  # rerun: nothing left to do
    assert "0 written" in capsys.readouterr().out


def test_total_rewards_in_summary_win(tmp_path):
    logs, out = tmp_path / "main", str(tmp_path / "agents.csv.gz")
    s = old_episode(logs, "harvest", [{"p0": 1.0, "p1": 1.0}], {"final_info": {"totals": {"p0": 9, "p1": 9}}})
    s.write_text(json.dumps({**json.loads(s.read_text()), "total_rewards": {"p0": 5.0, "p1": 4.0}}))
    backfill_rewards.main([str(logs), "--jobs", "1"])
    assert not s.with_name("episode_0.rewards.json").exists()  # already has its score
    summarize_runs.main([str(logs), "--out", out, "--jobs", "1"])
    assert rows(out) == {("harvest", "p0"): ("5.0", "total"), ("harvest", "p1"): ("4.0", "total")}

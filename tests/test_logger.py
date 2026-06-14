import json
from testbed.logging_.episode_logger import EpisodeLogger


def test_logger_writes_jsonl_lines(tmp_path):
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="run1", episode=0)
    logger.log_step(game="beauty_contest", turn=0, agent_id="player_0",
                    system_prompt="s", user_prompt="u", completion="CHOICE: 33",
                    parsed_action=33, parse_retries=0, reward=0.0,
                    steering_spec_id="noop")
    logger.log_step(game="beauty_contest", turn=0, agent_id="player_1",
                    system_prompt="s", user_prompt="u", completion="CHOICE: 50",
                    parsed_action=50, parse_retries=1, reward=1.0,
                    steering_spec_id="noop")
    logger.close(summary={"winner": "player_1"})

    path = tmp_path / "run1" / "episode_0.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["agent_id"] == "player_0"
    assert rec["parsed_action"] == 33

    summary = json.loads((tmp_path / "run1" / "episode_0.summary.json").read_text(encoding="utf-8"))
    assert summary["winner"] == "player_1"

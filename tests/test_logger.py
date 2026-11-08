import json
from testbed.logging_.episode_logger import EpisodeLogger


def test_logger_writes_jsonl_lines(tmp_path):
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="run1", episode=0)
    logger.log_step(game="beauty_contest", turn=0, agent_id="player_0",
                    system_prompt="s", user_prompt="u", completion="CHOICE: 33",
                    parsed_action=33, parse_retries=0, reward=0.0,
                    steering=None)
    logger.log_step(game="beauty_contest", turn=0, agent_id="player_1",
                    system_prompt="s", user_prompt="u", completion="CHOICE: 50",
                    parsed_action=50, parse_retries=1, reward=1.0,
                    steering=None)
    logger.close(summary={"winner": "player_1"})

    path = tmp_path / "run1" / "episode_0.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["agent_id"] == "player_0"
    assert rec["parsed_action"] == 33

    summary = json.loads((tmp_path / "run1" / "episode_0.summary.json").read_text(encoding="utf-8"))
    assert summary["winner"] == "player_1"


def _step(logger, agent, action):
    logger.log_step(game="beauty_contest", turn=0, agent_id=agent, system_prompt="s", user_prompt="u",
                    completion=f"CHOICE: {action}", parsed_action=action, parse_retries=0, reward=0.0, steering=None)


def test_two_writers_of_one_episode_leave_a_whole_file(tmp_path):
    """Two jobs running the same episode (a resubmit while the first still runs) used to share one open file and
    interleave or truncate it. Each writer now has its own part file, renamed into place on close."""
    a = EpisodeLogger(run_dir=str(tmp_path), run_id="run1", episode=0)
    b = EpisodeLogger(run_dir=str(tmp_path), run_id="run1", episode=0)
    _step(a, "player_0", 1)
    _step(b, "player_0", 2)
    _step(a, "player_1", 3)
    assert not (tmp_path / "run1" / "episode_0.jsonl").exists()  # in progress: no half-written episode is visible
    a.close(summary={"w": "a"})
    lines = (tmp_path / "run1" / "episode_0.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["parsed_action"] for l in lines] == [1, 3]
    _step(b, "player_1", 4)
    b.close(summary={"w": "b"})
    lines = (tmp_path / "run1" / "episode_0.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["parsed_action"] for l in lines] == [2, 4]  # last writer wins, whole
    assert not list((tmp_path / "run1").glob("*.part"))
    assert "CHOICE: 4" in (tmp_path / "run1" / "episode_0.trace.txt").read_text(encoding="utf-8")

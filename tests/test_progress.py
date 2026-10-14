from scripts.progress import main


def test_progress_reads_task_logs(tmp_path, capsys):
    logs, configs = tmp_path / "slurm-logs", tmp_path / "runs"
    logs.mkdir()
    (logs / "games-7_0.out").write_text(
        "[01:00:00] 3 run(s), 4 episode(s) to play in shard 0/2 of config/runs/main/gbs_3p\n"
        "[01:05:00] gbs_3p__none episode 0 done after 9 turns: {}\n"
        "[01:06:00] gbs_3p__one__evil_x1 episode 0 done after 9 turns: {}\n"
        "[01:07:00] ERROR gbs_3p__all__evil_x1 episode 0 (turn 2):\nTraceback ...\n\n")
    (logs / "games-7_1.out").write_text("loading engines...\n")  # started, not yet playing
    for group, n in (("gbs_3p", 3), ("werewolf_8p", 2)):
        (configs / group).mkdir(parents=True)
        for i in range(n):
            (configs / group / f"r{i}.yaml").write_text(f"run_id: r{i}\nepisodes: 2\n")
    assert main(["--logs", str(logs), "--configs", str(configs), "--tasks"]) == 0
    out = capsys.readouterr().out
    assert "2 done    1 failed of     4" in out
    assert "werewolf_8p" in out  # queued game still listed
    assert "all: 2 done, 1 failed of 10 episodes (30.0%)" in out


def test_errors_grouped(tmp_path, capsys):
    logs = tmp_path / "slurm-logs"
    logs.mkdir()
    tb = "Traceback (most recent call last):\n  File \"x.py\", line 1\nKeyError: 'phase'\n\n"
    (logs / "games-7_0.out").write_text(
        "[01:00:00] ERROR avalon_6p__one__evil_x1 episode 3 (turn 4):\n" + tb +
        "[01:01:00] ERROR avalon_6p__all__evil_x2 episode 5 (turn 9):\n" + tb +
        "[01:02:00] ERROR gbs_3p__none episode 0 (turn 0):\nTraceback ...\nValueError: bad\n\n")
    (logs / "games-7_1.out").write_text("vllm.v1.engine.exceptions.EngineDeadError: EngineCore died\n")
    assert main(["--logs", str(logs), "--errors"]) == 0
    out = capsys.readouterr().out
    assert "[2x] KeyError: 'phase'" in out and "avalon_6p (2)" in out
    assert "[1x] ValueError: bad" in out
    assert "EngineDeadError" in out and "7_1" in out

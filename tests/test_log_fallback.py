"""Episode logs fall back to MA_FALLBACK_LOG_DIR when the primary log dir fails; resume and the
summary see both."""
import errno
import json

from scripts.summarize_runs import agent_rows
from testbed.logging_ import episode_logger as L


def test_logger_falls_back_and_skips_the_failed_dir(tmp_path, monkeypatch):
    primary, fallback = tmp_path / "scratch", tmp_path / "fallback"
    real_makedirs = L.os.makedirs

    def broken(path, *a, **k):  # the primary filesystem is down
        if str(path).startswith(str(primary)):
            raise OSError(errno.EIO, "Communication error on send", str(path))
        return real_makedirs(path, *a, **k)

    monkeypatch.setattr(L.os, "makedirs", broken)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L, "_DOWN_UNTIL", {})
    log = L.EpisodeLogger(str(primary), "run_a", 0, fallback_dir=str(fallback), run_info={"run_id": "run_a"})
    log.close(summary={"final_rewards": {"p0": 1.0}})
    assert (fallback / "run_a" / "episode_0.summary.json").exists()
    assert json.loads((fallback / "run_a" / "run.json").read_text())["run_id"] == "run_a"
    assert str(primary) in L._DOWN_UNTIL  # next episodes skip the primary without waiting

    calls = []
    monkeypatch.setattr(L.os, "makedirs", lambda p, *a, **k: calls.append(str(p)) or real_makedirs(p, *a, **k))
    L.EpisodeLogger(str(primary), "run_a", 1, fallback_dir=str(fallback)).close()
    assert all(not c.startswith(str(primary)) for c in calls)


def test_summary_merges_dirs_first_wins(tmp_path):
    run = {"run_id": "g_2p__none", "game": {"id": "g", "players": 2}}
    for root, eps in (("a", (0, 1)), ("b", (1, 2))):
        d = tmp_path / root / "g_2p__none"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(run))
        for ep in eps:
            (d / f"episode_{ep}.summary.json").write_text(json.dumps({"final_rewards": {"p0": float(ord(root))}}))
    rows = list(agent_rows([tmp_path / "a" / "g_2p__none", tmp_path / "b" / "g_2p__none"]))
    assert [(r["episode"], r["reward"]) for r in rows] == [(0, 97.0), (1, 97.0), (2, 98.0)]

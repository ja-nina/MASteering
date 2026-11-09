"""testbed/pool.py: own shard first, then other shards' episodes, with claim files so no episode is played twice."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from testbed import pool as pool_mod
from testbed.pool import EpisodePool


def _job(ep, shard=0, run="r"):
    return SimpleNamespace(run_id=run, run_index=0, episode=ep, shard=shard)


def _pool(tmp_path, own, others=(), **kw):
    kw.setdefault("busy", lambda shard: False)
    return EpisodePool(own, others, run_dirs=lambda job: [tmp_path / job.run_id], **kw)


def _drain(pool):
    out = []
    while (job := pool.next()) is not None:
        out.append(job.episode)
    return out


def test_own_first_then_others(tmp_path):
    p = _pool(tmp_path, [_job(0), _job(1)], [_job(5, 1), _job(6, 2)])
    assert _drain(p) == [0, 1, 5, 6] and p.taken == 2


def test_a_claimed_episode_is_not_started_by_another_task(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_mod, "task_alive", lambda task: True)
    a = _pool(tmp_path, [_job(0)])
    a.task = "1_0"
    assert _drain(a) == [0] and (tmp_path / "r" / "episode_0.claim").exists()
    b = _pool(tmp_path, [], [_job(0, 0)])
    b.task = "1_1"
    assert _drain(b) == []  # task 1_0 is running it
    a.finished(SimpleNamespace(**{**vars(_job(0)), "_claim": tmp_path / "r" / "episode_0.claim"}), 10.0)
    assert not (tmp_path / "r" / "episode_0.claim").exists()


def test_claim_of_a_dead_task_is_stale(tmp_path, monkeypatch):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "episode_3.claim").write_text(json.dumps({"task": "9_9", "pid": 1, "time": time.time()}))
    monkeypatch.setattr(pool_mod, "task_alive", lambda task: False)
    assert _drain(_pool(tmp_path, [_job(3)])) == [3]
    claim = json.loads((tmp_path / "r" / "episode_3.claim").read_text())
    assert claim["pid"] == os.getpid()  # now ours


def test_old_claim_is_stale_when_squeue_cannot_tell(tmp_path, monkeypatch):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "episode_3.claim").write_text(json.dumps({"task": "9_9", "pid": 1, "time": time.time() - 1e6}))
    (tmp_path / "r" / "episode_4.claim").write_text(json.dumps({"task": "9_9", "pid": 1, "time": time.time()}))
    monkeypatch.setattr(pool_mod, "task_alive", lambda task: None)
    assert _drain(_pool(tmp_path, [_job(3), _job(4)])) == [3]


def test_done_episodes_and_busy_shards_are_skipped(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "episode_1.summary.json").write_text("{}")
    p = _pool(tmp_path, [_job(0), _job(1)], [_job(5, 1), _job(6, 2)], busy=lambda shard: shard == 1)
    assert _drain(p) == [0, 6]


def test_others_only_with_time_for_one_more_episode(tmp_path):
    p = _pool(tmp_path, [_job(0)], [_job(5, 1), _job(6, 1)], deadline=time.time() + 1000)
    assert _drain(p) == [0]  # nothing finished yet: unknown how long an episode takes
    p.finished(_job(0), 500.0)  # 1.25 x 500 < 1000 s left
    assert p.next().episode == 5
    p.finished(_job(5), 900.0)  # 1.25 x 900 > 1000
    assert p.next() is None


def test_unwritable_claim_dir_still_plays_own_but_takes_nothing(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise PermissionError("read-only")
    monkeypatch.setattr(pool_mod.os, "open", broken)
    assert _drain(_pool(tmp_path, [_job(0)], [_job(5, 1)])) == [0]


def test_shard_busy_reads_the_array_once(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="0 RUNNING\n1 PENDING\n2 RUNNING\n", stderr="")
    monkeypatch.setattr(pool_mod.subprocess, "run", run)
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "777")
    pool_mod._array_states.cache_clear()
    assert [pool_mod.shard_busy(s) for s in range(4)] == [True, False, True, False]
    assert len(calls) == 1 and "-r" in calls[0]


def test_first_is_handed_out_by_next(tmp_path):
    p = _pool(tmp_path, [_job(0), _job(1)])
    assert p.first().episode == 0
    assert _drain(p) == [0, 1]  # the job first() claimed is not lost or claimed twice
    assert _pool(tmp_path / "x", []).first() is None

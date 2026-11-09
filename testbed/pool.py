"""Which episode a task plays next: its own shard first, then unstarted episodes of the other shards.

A task of an array plays its shard (config.build_jobs). With 50 episodes per task and 64 concurrent, every
episode starts at once and slots go idle as episodes end. EpisodePool refills them: once a slot frees, the task
takes an episode of another shard that nobody has started (from the end of that shard's list: the episodes its
own task would reach last), if there is time left to finish one. Two tasks never play the same episode:

  claim   <run_dir>/episode_N.claim, created exclusively (O_EXCL) before an episode starts, by its own task as
          well as by a taker, and removed when the episode ends (done or failed). An existing claim is honoured
          unless it is stale: its SLURM job is no longer running (squeue), or, where squeue cannot tell, it is
          older than MA_CLAIM_TTL_S (default 9 h, above the 8 h time limit).
  done    an episode with a summary file (logging dir or fallback) is never started again.
  busy    episodes of a shard whose own task is running are not taken: it is playing or about to play them (and a
          task started before the pool existed writes no claims). Taken are the shards of tasks still pending, or
          ended (timed out, died) with episodes left: one squeue call for the whole array, every 5 min.
  time    an own episode after the first finished one, and another shard's episode at all, is started only when the task's remaining time (SLURM end time) exceeds
          MA_TAKE_MARGIN (default 1.25) x the longest episode this task has finished, so a taken episode is
          not cut off by the time limit (a cut-off episode is lost: nothing of it is kept).

Which episodes are played, and how, does not change: the same jobs (seeds, steering) as their own shard would
have played; only which task plays them. A task that starts after its episodes were taken finds them done or
claimed and goes on to take others'.
"""
from __future__ import annotations

import functools
import json
import os
import re
import socket
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional

from testbed.util import log

CLAIM_TTL_S = float(os.environ.get("MA_CLAIM_TTL_S", 9 * 3600))
TAKE_MARGIN = float(os.environ.get("MA_TAKE_MARGIN", 1.25))


def _task_id() -> str:
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}"
    return os.environ.get("SLURM_JOB_ID", "")


def slurm_deadline() -> Optional[float]:
    """When this SLURM job is killed (epoch seconds), from squeue's time left; None outside SLURM."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", left)
    if not m:
        return None
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return time.time() + ((d * 24 + h) * 60 + mi) * 60 + s


@functools.lru_cache(maxsize=4096)
def _alive_cached(task: str, minute: int) -> Optional[bool]:
    try:
        out = subprocess.run(["squeue", "-h", "-j", task, "-o", "%T"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 and "Invalid job id" not in (out.stderr or ""):
        return None
    return bool(out.stdout.strip()) and out.stdout.split()[0] in ("RUNNING", "COMPLETING", "PENDING", "CONFIGURING")


@functools.lru_cache(maxsize=64)
def _array_states(array_job: str, minute: int) -> Optional[dict]:
    try:
        out = subprocess.run(["squeue", "-h", "-r", "-j", array_job, "-o", "%K %T"], capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return {} if "Invalid job id" in (out.stderr or "") else None
    states = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            states[int(parts[0])] = parts[1]
    return states


def shard_busy(shard: int) -> bool:
    """Is the array task that owns this shard running? (Our own array; False outside SLURM arrays.)"""
    array = os.environ.get("SLURM_ARRAY_JOB_ID")
    if not array:
        return False
    states = _array_states(array, int(time.time() // 300))
    if states is None:
        return True  # squeue cannot tell: take nothing rather than risk playing an episode twice
    return states.get(shard) in ("RUNNING", "COMPLETING", "CONFIGURING")


def task_alive(task: str) -> Optional[bool]:
    """Is this SLURM job (or array task "JOB_IDX") still running? None when squeue cannot tell."""
    if not task:
        return None
    return _alive_cached(task, int(time.time() // 300))  # asked again at most every 5 min


class EpisodePool:
    """Hands out jobs to run_jobs: next() and finished(). own: this task's shard; others: every other shard's
    unfinished episodes, the ones to take first first. run_dirs(job) -> [primary dir, fallback dir, ...]."""

    def __init__(self, own: Iterable, others: Iterable = (), run_dirs=None, deadline: Optional[float] = None,
                 claims: bool = True, busy=shard_busy) -> None:
        self.own, self.others = deque(own), deque(others)
        self.busy = busy
        self.run_dirs = run_dirs or (lambda job: [])
        self.deadline, self.claims = deadline, claims
        self.durations: List[float] = []
        self.taken = 0
        self.task = _task_id()

    def __bool__(self) -> bool:
        return bool(self.own or self.others)

    # ---- the files
    def _claim_path(self, job) -> Optional[Path]:
        dirs = self.run_dirs(job)
        return Path(dirs[0]) / f"episode_{job.episode}.claim" if dirs else None

    def _done(self, job) -> bool:
        for d in self.run_dirs(job):
            try:
                if (Path(d) / f"episode_{job.episode}.summary.json").exists():
                    return True
            except OSError:
                pass
        return False

    def _claim(self, job) -> bool:
        """True when this task may play the job (claimed now, or claims are off)."""
        path = self._claim_path(job)
        if not self.claims or path is None:
            return True
        for _ in range(2):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._stale(path):
                    return False
                try:
                    path.unlink()
                except OSError:
                    return False
                continue
            except OSError:
                return None  # the file system cannot tell: the caller decides
            with os.fdopen(fd, "w") as f:
                json.dump({"task": self.task, "host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}, f)
            job._claim = path
            return True
        return False

    def _stale(self, path: Path) -> bool:
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):  # half-written or unreadable: judge by age
            try:
                return time.time() - path.stat().st_mtime > CLAIM_TTL_S
            except OSError:
                return False
        if info.get("task") and info["task"] == self.task and info.get("pid") != os.getpid():
            return True  # this task's earlier process (a restart): gone
        alive = task_alive(info.get("task", ""))
        if alive is not None:
            return not alive
        return time.time() - float(info.get("time", 0)) > CLAIM_TTL_S

    def _release(self, job) -> None:
        path = getattr(job, "_claim", None)
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass
            job._claim = None

    # ---- run_jobs' side
    def _time_for_one_more(self) -> bool:
        if self.deadline is None:
            return True
        if not self.durations:
            return False  # nothing finished yet: no idea how long an episode takes
        return time.time() + TAKE_MARGIN * max(self.durations) < self.deadline

    def first(self):
        """next(), kept for the next call to next(): whether there is anything to play, before engines load."""
        self._first = self.next()
        return self._first

    def next(self):
        """The next job to start, or None (nothing left that can be started now)."""
        if getattr(self, "_first", None) is not None:
            job, self._first = self._first, None
            return job
        # own episodes too: once episodes have finished (so their length is known), one starts only if it can end
        # before the time limit; the ones left are taken by other tasks once this one has ended, or resubmitted
        while self.own and (not self.durations or self._time_for_one_more()):
            job = self.own.popleft()
            if self._done(job):
                continue
            ok = self._claim(job)
            if ok is False:
                continue  # another task took it
            return job  # claimed, or the file system could not tell: play it (as without the pool)
        while self.others and self._time_for_one_more():
            job = self.others.popleft()
            if self.busy(getattr(job, "shard", -1)) or self._done(job) or self._claim(job) is not True:
                continue  # its own task plays it, it is done, or another task took it
            self.taken += 1
            log(f"taking over {job.run_id} episode {job.episode} (another shard's; {self.taken} so far)")
            return job
        return None

    def finished(self, job, seconds: Optional[float]) -> None:
        if seconds is not None:
            self.durations.append(seconds)
        self._release(job)


def pool_from_lists(own: list, others: list = ()) -> EpisodePool:
    """A pool without claim files (tests, local runs): own first, then others."""
    return EpisodePool(own, others, claims=False, busy=lambda shard: False)

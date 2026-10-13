"""Melting Pot substrates behind a small interface, in-process or in a worker process.

    action_names: list[str]
    reset() -> list[obs]                       one dict per player: "RGB" (H, W, 3) uint8, scalars, ...
    step(actions: list[int]) -> (list[obs], list[float] rewards, done)
    close()

Melting Pot needs dmlab2d (Linux, Python >= 3.11, glibc >= 2.35 for the 3.12 wheels). When it
cannot be installed in the vLLM env, give the adapter `python=/path/to/other/env/python` (or set
MELTINGPOT_PYTHON): the substrate then runs in a worker process started with that interpreter,
which only needs numpy + dm-meltingpot and this repo on its path.

Substrate names starting with "fake:" use FakeSubstrate (tests; no Melting Pot needed).
"""
from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_MOVES = {1: "FORWARD", 2: "STEP_RIGHT", 3: "BACKWARD", 4: "STEP_LEFT"}
_TURNS = {-1: "TURN_LEFT", 1: "TURN_RIGHT"}


def action_name(action: dict) -> str:
    """Name of a Melting Pot action dict, e.g. {"move": 0, "turn": 0, "fireZap": 1} -> FIRE_ZAP."""
    parts = []
    for key, value in action.items():
        if not value:
            continue
        if key == "move":
            parts.append(_MOVES.get(value, f"MOVE_{value}"))
        elif key == "turn":
            parts.append(_TURNS.get(value, f"TURN_{value}"))
        else:
            parts.append(re.sub(r"(?<!^)(?=[A-Z])", "_", key).upper() + ("" if value == 1 else f"_{value}"))
    return "+".join(parts) or "NOOP"


def _plain(obs) -> dict:
    """numpy arrays stay arrays (RGB views); size-1 arrays become floats."""
    out = {}
    for k, v in dict(obs).items():
        a = np.asarray(v)
        out[k] = float(a.reshape(-1)[0]) if a.size == 1 else a
    return out


def _hide_gpus_from_tensorflow() -> None:
    """meltingpot imports tensorflow; keep it off the GPUs vLLM runs on (substrates need no TF ops)."""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    except Exception:  # no tensorflow, or devices already initialised
        pass


class LocalSubstrate:
    def __init__(self, name: str, num_players: int) -> None:
        _hide_gpus_from_tensorflow()
        from meltingpot import substrate

        config = substrate.get_config(name)
        default = tuple(config.default_player_roles)
        roles = default if num_players == len(default) else tuple(default[i % len(default)] for i in range(num_players))
        self.roles = list(roles)
        self.action_names = [action_name(dict(a)) for a in config.action_set]
        self.env = substrate.build(name, roles=roles)

    def reset(self) -> List[dict]:
        return [_plain(o) for o in self.env.reset().observation]

    def step(self, actions: List[int]) -> Tuple[List[dict], List[float], bool]:
        ts = self.env.step(actions)
        return [_plain(o) for o in ts.observation], [float(r) for r in ts.reward], bool(ts.last())

    def close(self) -> None:
        self.env.close()


class FakeSubstrate:
    """'fake:<steps>': 2-colour 40x40 views, reward 1 for FORWARD, done after <steps> steps."""

    def __init__(self, name: str, num_players: int) -> None:
        self.n, self.limit = num_players, int(name.split(":")[1]) if ":" in name and name.split(":")[1] else 50
        self.roles = ["default"] * num_players
        self.action_names = ["NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT",
                             "FIRE_ZAP"]
        self.t = 0

    def _obs(self) -> List[dict]:
        rgb = np.zeros((40, 40, 3), np.uint8)
        rgb[: 4 + self.t % 30] = (60, 160, 60)
        return [{"RGB": rgb.copy(), "READY_TO_SHOOT": 1.0, "WORLD.RGB": np.zeros((48, 64, 3), np.uint8)}
                for _ in range(self.n)]

    def reset(self) -> List[dict]:
        self.t = 0
        return self._obs()

    def step(self, actions: List[int]):
        self.t += 1
        return self._obs(), [float(a == 1) for a in actions], self.t >= self.limit

    def close(self) -> None:
        pass


class WorkerSubstrate:
    """The same interface, served by `python -m testbed.envs.meltingpot.backend` in another interpreter."""

    def __init__(self, name: str, num_players: int, python: str) -> None:
        root = str(Path(__file__).resolve().parents[3])
        env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        self.proc = subprocess.Popen([python, "-m", "testbed.envs.meltingpot.backend", name, str(num_players)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=root, env=env)
        self.action_names, self.roles = self._recv()

    def _recv(self):
        try:
            result = pickle.load(self.proc.stdout)
        except EOFError:
            raise RuntimeError(f"Melting Pot worker exited (code {self.proc.wait()}); see its stderr") from None
        if isinstance(result, BaseException):
            raise result
        return result

    def _call(self, cmd, arg=None):
        pickle.dump((cmd, arg), self.proc.stdin)
        self.proc.stdin.flush()
        return self._recv()

    def reset(self):
        return self._call("reset")

    def step(self, actions):
        return self._call("step", list(actions))

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._call("close")
            finally:
                self.proc.wait(timeout=30)


def make_substrate(name: str, num_players: int, python: str = None):
    python = python or os.environ.get("MELTINGPOT_PYTHON")
    if python:
        return WorkerSubstrate(name, num_players, python)
    return (FakeSubstrate if name.startswith("fake") else LocalSubstrate)(name, num_players)


def _serve(name: str, num_players: int) -> None:
    out = os.fdopen(os.dup(1), "wb")  # the protocol channel; anything printed goes to stderr instead
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        pickle.dump(obj, out)
        out.flush()

    try:
        sub = (FakeSubstrate if name.startswith("fake") else LocalSubstrate)(name, num_players)
    except Exception as e:  # noqa: BLE001 - reported to the parent, which raises it
        send(e)
        return
    send((sub.action_names, sub.roles))
    while True:
        try:
            cmd, arg = pickle.load(sys.stdin.buffer)
        except EOFError:
            break
        try:
            if cmd == "reset":
                send(sub.reset())
            elif cmd == "step":
                send(sub.step(arg))
            elif cmd == "close":
                sub.close()
                send(None)
                break
        except Exception as e:  # noqa: BLE001
            send(e)


if __name__ == "__main__":
    _serve(sys.argv[1], int(sys.argv[2]))

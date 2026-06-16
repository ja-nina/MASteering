"""Unit tests for the disk-bounded layer-sweep orchestrator. All subprocess
calls are replaced with a recording fake — no real model, GPU, or training
runs in this test."""
import argparse
import os

from scripts.run_layer_sweep import run_sweep, _combined_path


def _make_args(tmp_path, **overrides):
    defaults = dict(
        game="beauty_contest", model="fake-model", streams=["resid", "mlp"],
        start_layer=10, end_layer=11, num_episodes=5, max_rounds=4, num_players=4,
        seed=42, activations_dir=str(tmp_path / "activations"),
        sae_dir=str(tmp_path / "sae"), vectors_dir=str(tmp_path / "vectors"),
        d_sae=16384, k=32, epochs=20, top_n=16, wandb=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_sweep_deletes_combined_activations_after_each_layer(tmp_path):
    args = _make_args(tmp_path)
    os.makedirs(args.activations_dir, exist_ok=True)
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        if cmd[1].endswith("collect_activations.py"):
            layer = cmd[cmd.index("--layer") + 1]
            for stream in args.streams:
                path = _combined_path(args.activations_dir, args.game, layer, stream)
                open(path, "w").close()
        return None

    run_sweep(args, run=fake_run)

    # 2 layers x (1 collect + 2 streams x 2 (train_sae + find_tom_features)) = 10
    assert len(calls) == 10
    for layer_idx in (10, 11):
        for stream in args.streams:
            path = _combined_path(args.activations_dir, args.game,
                                  f"model.layers.{layer_idx}", stream)
            assert not os.path.exists(path)


def test_run_sweep_auto_detects_end_layer(monkeypatch, tmp_path):
    args = _make_args(tmp_path, end_layer=None)
    monkeypatch.setattr("scripts.run_layer_sweep._detect_end_layer", lambda model: 10)
    calls = []
    run_sweep(args, run=lambda cmd, check=True: calls.append(cmd))
    # start=10, detected end=10 -> exactly one layer's worth: 1 + 2*2 = 5
    assert len(calls) == 5

"""Sweep every layer's residual stream and MLP output through the full
extract -> train SAE -> find ToM vector pipeline, one (layer, stream) combo
at a time, deleting each combo's large raw-activation file as soon as its
SAE and steering vector are produced.

This keeps peak scratch usage to roughly one layer's worth of raw
activations (~90GB at Qwen3-4B's size for both streams) instead of
accumulating every layer simultaneously (multi-TB). The trade-off: the
model's forward pass is re-run once per swept layer instead of once total,
since a forward pass always computes every layer regardless of which one is
hooked. See docs/superpowers/specs/2026-06-16-qwen3-4b-layer-sweep-design.md.

Usage
-----
python scripts/run_layer_sweep.py \\
    --game beauty_contest --model Qwen/Qwen3-4B \\
    --start-layer 10 --num-episodes 200 --max-rounds 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _detect_end_layer(model_id: str) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return cfg.num_hidden_layers - 1


def _layer_tag(layer: str) -> str:
    return layer.replace(".", "_")


def _combined_path(activations_dir: str, game: str, layer: str, stream: str) -> str:
    return os.path.join(activations_dir, f"base_{game}_{_layer_tag(layer)}_{stream}.npy")


def _base_last_path(activations_dir: str, game: str, layer: str, stream: str) -> str:
    return os.path.join(activations_dir, f"base_last_{game}_{_layer_tag(layer)}_{stream}.npy")


def _tom_last_path(activations_dir: str, game: str, layer: str, stream: str) -> str:
    return os.path.join(activations_dir, f"tom_{game}_{_layer_tag(layer)}_{stream}.npy")


def _sae_path(sae_dir: str, game: str, layer: str, stream: str, d_sae: int, k: int) -> str:
    return os.path.join(sae_dir, f"{game}_{_layer_tag(layer)}_{stream}_d{d_sae}_k{k}.pt")


def _collect_command(args, layer: str) -> list[str]:
    return [sys.executable, "scripts/collect_activations.py",
            "--game", args.game, "--model", args.model, "--layer", layer,
            "--streams", ",".join(args.streams),
            "--num-episodes", str(args.num_episodes),
            "--max-rounds", str(args.max_rounds),
            "--num-players", str(args.num_players),
            "--output-dir", args.activations_dir,
            "--seed", str(args.seed)]


def _train_sae_command(args, layer: str, stream: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_sae.py",
           "--activations", _combined_path(args.activations_dir, args.game, layer, stream),
           "--d-sae", str(args.d_sae), "--k", str(args.k), "--epochs", str(args.epochs),
           "--output", _sae_path(args.sae_dir, args.game, layer, stream, args.d_sae, args.k)]
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def _find_tom_features_command(args, layer: str, stream: str) -> list[str]:
    cmd = [sys.executable, "scripts/find_tom_features.py",
           "--sae", _sae_path(args.sae_dir, args.game, layer, stream, args.d_sae, args.k),
           "--base", _base_last_path(args.activations_dir, args.game, layer, stream),
           "--tom", _tom_last_path(args.activations_dir, args.game, layer, stream),
           "--top-n", str(args.top_n), "--output-dir", args.vectors_dir]
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def run_sweep(args, run=subprocess.run) -> None:
    os.makedirs(args.activations_dir, exist_ok=True)
    os.makedirs(args.sae_dir, exist_ok=True)
    os.makedirs(args.vectors_dir, exist_ok=True)

    end_layer = args.end_layer
    if end_layer is None:
        end_layer = _detect_end_layer(args.model)
        print(f"Auto-detected end layer: {end_layer}")

    for layer_idx in range(args.start_layer, end_layer + 1):
        layer = f"model.layers.{layer_idx}"
        print(f"\n=== layer {layer_idx} ({layer}) ===")

        run(_collect_command(args, layer), check=True)

        for stream in args.streams:
            run(_train_sae_command(args, layer, stream), check=True)
            run(_find_tom_features_command(args, layer, stream), check=True)

            combined = _combined_path(args.activations_dir, args.game, layer, stream)
            if os.path.exists(combined):
                os.remove(combined)
                print(f"  deleted {combined}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="beauty_contest", choices=["beauty_contest", "gbs"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--streams", default="resid,mlp")
    ap.add_argument("--start-layer", type=int, default=10)
    ap.add_argument("--end-layer", type=int, default=None,
                    help="Default: auto-detect (model's num_hidden_layers - 1)")
    ap.add_argument("--num-episodes", type=int, default=500)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--activations-dir", default="/scratch/inf0/user/nzukowsk/activations")
    ap.add_argument("--sae-dir", default="/scratch/inf0/user/nzukowsk/sae")
    ap.add_argument("--vectors-dir", default="vectors")
    ap.add_argument("--d-sae", type=int, default=16384)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--wandb", action="store_true",
                    help="Forward --wandb to train_sae.py / find_tom_features.py")
    args = ap.parse_args()
    args.streams = args.streams.split(",")
    run_sweep(args)


if __name__ == "__main__":
    main()

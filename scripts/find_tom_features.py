"""Find ToM-relevant SAE features and export them as steering vectors.

Loads a trained SAE and paired (base, ToM-suffix) last-token activations,
encodes both through the SAE, and scores each feature by:

    delta_i = mean(f_i | ToM prompts) - mean(f_i | base prompts)

Features with the highest positive delta fire more under the ToM suffix
and represent the model's "other-agent reasoning" direction in SAE space.

Exports:
  vectors/tom_sae_top{N}_{stem}.npy   — steering vector reconstructed from
                                         top-N feature decoder columns
  vectors/tom_sae_features_{stem}.csv — feature scores for inspection

Usage
-----
python scripts/find_tom_features.py \\
    --sae sae/beauty_contest_l18_4096k32.pt \\
    --base activations/base_last_beauty_contest_model_layers_18.npy \\
    --tom  activations/tom_beauty_contest_model_layers_18.npy \\
    --top-n 16

Then set in run_config.yaml:
    steering:
      default: activation
      per_agent:
        player_0:
          layer: model.layers.18
          vector_path: vectors/tom_sae_top16_beauty_contest_l18_4096k32.npy
          coefficient: 20.0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── SAE (must match train_sae.py) ────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int) -> None:
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae
        self.k     = k
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts


def load_sae(path: str) -> tuple[TopKSAE, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg  = ckpt["config"]
    sae  = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt["norm"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae",    required=True, help="SAE .pt from train_sae.py")
    ap.add_argument("--base",   required=True,
                    help="base_last_*.npy from collect_activations.py")
    ap.add_argument("--tom",    required=True,
                    help="tom_*.npy from collect_activations.py")
    ap.add_argument("--top-n",  type=int, default=16,
                    help="Number of top features to include in steering vector")
    ap.add_argument("--output-dir", default="vectors")
    ap.add_argument("--wandb", action="store_true",
                    help="Log top feature scores and steering vector stats to wandb")
    ap.add_argument("--wandb-project", default="ma-steering-sae")
    ap.add_argument("--wandb-name", default=None,
                    help="Defaults to the SAE checkpoint stem")
    args = ap.parse_args()

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed — skipping. pip install wandb to enable.")
        else:
            sae_stem_for_name = os.path.splitext(os.path.basename(args.sae))[0]
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"tom_features_{sae_stem_for_name}",
                config={"sae": args.sae, "base": args.base, "tom": args.tom,
                        "top_n": args.top_n})

    os.makedirs(args.output_dir, exist_ok=True)
    sae_stem = os.path.splitext(os.path.basename(args.sae))[0]

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"Loading SAE from {args.sae} ...")
    sae, norm = load_sae(args.sae)
    mean = torch.from_numpy(norm["mean"]).float()
    std  = torch.from_numpy(norm["std"]).float()

    print(f"Loading activations ...")
    base_np = np.load(args.base).astype(np.float32)
    tom_np  = np.load(args.tom).astype(np.float32)
    assert base_np.shape == tom_np.shape, \
        f"Shape mismatch: base {base_np.shape} vs tom {tom_np.shape}"
    print(f"  paired samples: {base_np.shape[0]}")

    # normalise with same stats used during SAE training
    base_t = (torch.from_numpy(base_np) - mean) / (std + 1e-8)
    tom_t  = (torch.from_numpy(tom_np)  - mean) / (std + 1e-8)

    # ── encode ────────────────────────────────────────────────────────────────
    print("Encoding through SAE ...")
    with torch.no_grad():
        base_acts = sae.encode(base_t)   # [P, d_sae]
        tom_acts  = sae.encode(tom_t)    # [P, d_sae]

    # ── score features ────────────────────────────────────────────────────────
    delta = (tom_acts - base_acts).mean(dim=0)   # [d_sae]  positive = ToM-activating
    scores, ranked = delta.sort(descending=True)

    print(f"\nTop-{args.top_n} ToM-activating features:")
    print(f"  {'rank':>4}  {'feat_id':>7}  {'delta':>10}  "
          f"{'mean_base':>10}  {'mean_tom':>10}")
    for rank in range(args.top_n):
        fid = ranked[rank].item()
        print(f"  {rank+1:>4}  {fid:>7}  {scores[rank].item():>10.4f}  "
              f"{base_acts[:, fid].mean().item():>10.4f}  "
              f"{tom_acts[:, fid].mean().item():>10.4f}")

    # ── build steering vector from top-N decoder columns ─────────────────────
    # decoder weight: [d_in, d_sae] (PyTorch Linear stores W^T)
    decoder_cols = sae.decoder.weight.data  # [d_in, d_sae]
    top_ids = ranked[:args.top_n]
    top_scores = scores[:args.top_n]

    # weight each column by its delta score, sum → steering direction
    weighted = (decoder_cols[:, top_ids] * top_scores.unsqueeze(0)).sum(dim=1)
    # normalise to same scale as a unit CAA vector
    steering_vec = weighted / (weighted.norm() + 1e-8)

    out_vec = os.path.join(args.output_dir,
                           f"tom_sae_top{args.top_n}_{sae_stem}.npy")
    np.save(out_vec, steering_vec.numpy())
    print(f"\nSteering vector saved: {out_vec}")
    print(f"  shape={steering_vec.shape}  norm={steering_vec.norm().item():.4f}")

    # ── save CSV for inspection ───────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f"tom_features_{sae_stem}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature_id", "delta",
                    "mean_base_activation", "mean_tom_activation"])
        for rank in range(min(200, sae.d_sae)):  # save top 200 for inspection
            fid = ranked[rank].item()
            w.writerow([
                rank + 1, fid,
                f"{scores[rank].item():.6f}",
                f"{base_acts[:, fid].mean().item():.6f}",
                f"{tom_acts[:, fid].mean().item():.6f}",
            ])
    print(f"Feature scores saved: {csv_path}")

    if wandb_run is not None:
        table = wandb.Table(columns=["rank", "feature_id", "delta",
                                     "mean_base_activation", "mean_tom_activation"])
        for rank in range(min(200, sae.d_sae)):
            fid = ranked[rank].item()
            table.add_data(rank + 1, fid, scores[rank].item(),
                           base_acts[:, fid].mean().item(),
                           tom_acts[:, fid].mean().item())
        wandb_run.log({
            "tom_features": table,
            "steering_vector_norm": steering_vec.norm().item(),
            "top1_delta": scores[0].item(),
            "top_n_mean_delta": scores[:args.top_n].mean().item(),
        })
        wandb_run.finish()

    print()
    print("To apply in run_config.yaml:")
    print(f"  steering:")
    print(f"    default: activation")
    print(f"    per_agent:")
    print(f"      player_0:")
    print(f"        layer: model.layers.18   # match layer used in collect_activations.py")
    print(f"        vector_path: {out_vec}")
    print(f"        coefficient: 20.0        # tune: try 10, 20, 40")


if __name__ == "__main__":
    main()

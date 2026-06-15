"""Train a TopK Sparse Autoencoder on collected residual-stream activations.

TopK SAEs enforce sparsity by keeping only the k largest pre-activations
per sample — no L1 penalty needed. The decoder columns are normalised to
unit norm after each gradient step so feature magnitudes stay comparable.

Output: sae/<stem>.pt  containing the model state dict + config metadata.

Usage
-----
python scripts/train_sae.py \\
    --activations activations/base_beauty_contest_model_layers_18.npy \\
    --d-sae 4096 \\
    --k 32 \\
    --output sae/beauty_contest_l18_4096k32.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── model ─────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int) -> None:
        super().__init__()
        self.d_in  = d_in
        self.d_sae = d_sae
        self.k     = k
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        # initialise decoder columns to unit norm
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalise_decoder()

    @torch.no_grad()
    def _normalise_decoder(self):
        """Keep decoder columns (features) at unit norm."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_in] -> sparse feature activations [B, d_sae]."""
        pre = self.encoder(x)                         # [B, d_sae]
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        recon = self.decoder(acts)
        return recon, acts


# ── dataset ───────────────────────────────────────────────────────────────────

class _ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, acts: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.data = torch.from_numpy(
            (acts - mean) / (std + 1e-8)).float()

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ── training ──────────────────────────────────────────────────────────────────

def train(sae: TopKSAE, loader, epochs: int, lr: float, device: str):
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for ep in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        total_dead = torch.zeros(sae.d_sae, device=device)
        for batch in loader:
            batch = batch.to(device)
            recon, acts = sae(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae._normalise_decoder()
            total_loss += loss.item()
            total_dead += (acts.sum(0) == 0).float()

        dead_pct = (total_dead > 0).float().mean().item() * 100
        avg_loss = total_loss / len(loader)
        print(f"  epoch {ep+1:3d}/{epochs}  loss={avg_loss:.6f}  "
              f"dead_features={dead_pct:.1f}%  "
              f"({time.time()-t0:.1f}s)")
    return sae


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True,
                    help="base_*.npy file from collect_activations.py")
    ap.add_argument("--d-sae", type=int, default=4096,
                    help="SAE dictionary size (default: 4096)")
    ap.add_argument("--k", type=int, default=32,
                    help="TopK sparsity — active features per sample (default: 32)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # derive output path from input filename if not given
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.activations))[0]
        # strip leading "base_"
        stem = stem.removeprefix("base_")
        os.makedirs("sae", exist_ok=True)
        args.output = f"sae/{stem}_d{args.d_sae}_k{args.k}.pt"

    print(f"Loading activations from {args.activations} ...")
    acts = np.load(args.activations)
    print(f"  shape: {acts.shape}  dtype: {acts.dtype}")

    # normalise
    mean = acts.mean(axis=0)
    std  = acts.std(axis=0)
    d_in = acts.shape[1]

    dataset = _ActivationDataset(acts, mean, std)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"))

    print(f"\nTraining TopK-SAE  d_in={d_in}  d_sae={args.d_sae}  k={args.k}")
    print(f"  samples={len(dataset)}  batch={args.batch_size}  "
          f"epochs={args.epochs}  lr={args.lr}\n")

    sae = TopKSAE(d_in=d_in, d_sae=args.d_sae, k=args.k)
    sae = train(sae, loader, epochs=args.epochs, lr=args.lr, device=device)

    # save model + normalisation stats
    torch.save({
        "state_dict": sae.cpu().state_dict(),
        "config": {"d_in": d_in, "d_sae": args.d_sae, "k": args.k},
        "norm": {"mean": mean, "std": std},
    }, args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

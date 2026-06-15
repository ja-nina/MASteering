"""Train a TopK Sparse Autoencoder on collected residual-stream activations.

TopK SAEs enforce sparsity by keeping only the k largest pre-activations
per sample — no L1 penalty needed. The decoder columns are normalised to
unit norm after each gradient step so feature magnitudes stay comparable.

Dead-feature resampling: every `resample_interval` epochs, features that
never activated during that epoch have their encoder/decoder weights reset
to normalised directions of high-residual samples. This breaks the
rich-get-richer dynamic that otherwise leaves most features permanently dead.

Output: sae/<stem>.pt  containing the model state dict + config metadata.

Usage
-----
python scripts/train_sae.py \\
    --activations activations/base_beauty_contest_model_layers_18.npy \\
    --d-sae 16384 \\
    --k 32 \\
    --output sae/beauty_contest_l18_16384k32.pt
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
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalise_decoder()

    @torch.no_grad()
    def _normalise_decoder(self):
        """Keep decoder columns (features) at unit norm."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_in] -> sparse feature activations [B, d_sae]."""
        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        recon = self.decoder(acts)
        return recon, acts

    @torch.no_grad()
    def resample_dead(self, dead_mask: torch.Tensor,
                      data: torch.Tensor, noise: float = 0.01) -> int:
        """Reset dead feature weights to directions of high-residual samples.

        dead_mask: [d_sae] bool, True = never activated this epoch
        data:      [N, d_in] sample of training data (normalised), on same device
        Returns number of features resampled.
        """
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # find the samples the current SAE explains worst
        recon, _ = self.forward(data)
        residuals = (data - recon).norm(dim=-1)   # [N]
        # sample proportional to squared residual (high-error samples get priority)
        probs = (residuals ** 2)
        probs = probs / probs.sum()
        chosen = torch.multinomial(probs, int(n_dead), replacement=True)
        new_dirs = data[chosen]                    # [n_dead, d_in]
        new_dirs = F.normalize(new_dirs, dim=-1)

        dead_idx = dead_mask.nonzero(as_tuple=False).squeeze(1)

        # reset encoder rows: direction of the sample + small noise
        self.encoder.weight[dead_idx] = new_dirs + noise * torch.randn_like(new_dirs)
        self.encoder.bias[dead_idx]   = 0.0

        # reset decoder columns to same direction (unit norm by construction)
        self.decoder.weight[:, dead_idx] = new_dirs.T

        return int(n_dead)


# ── dataset ───────────────────────────────────────────────────────────────────

class _ActivationDataset(torch.utils.data.Dataset):
    """Reads activations row-by-row from a memmap — never loads full array."""
    def __init__(self, acts: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.acts = acts
        self.mean = mean.astype(np.float32)
        self.std  = (std + 1e-8).astype(np.float32)

    def __len__(self): return len(self.acts)

    def __getitem__(self, i):
        x = self.acts[i].astype(np.float32)
        return torch.from_numpy((x - self.mean) / self.std)


# ── training ──────────────────────────────────────────────────────────────────

def train(sae: TopKSAE, dataset, train_loader, val_loader,
          epochs: int, lr: float, device: str,
          resample_interval: int = 5, resample_until: int = 10,
          resample_samples: int = 8192):
    """
    resample_interval  — resample dead features every N epochs
    resample_until     — stop resampling after this epoch (default: 10,
                         i.e. first half of the default 20-epoch run)
    Resampling only fires when >1% of features were dead that epoch.
    """
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    # cache a fixed sample of data for residual-based resampling
    idx = torch.randperm(len(dataset))[:resample_samples]
    resample_data = torch.stack([dataset[i] for i in idx]).to(device)

    for ep in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        never_activated = torch.ones(sae.d_sae, dtype=torch.bool, device=device)

        sae.train()
        for batch in train_loader:
            batch = batch.to(device)
            recon, acts = sae(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae._normalise_decoder()
            total_loss += loss.item()
            never_activated &= (acts.sum(0) == 0)

        train_loss = total_loss / len(train_loader)
        dead_pct   = never_activated.float().mean().item() * 100

        # validation loss
        sae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = sae(batch)
                val_loss += F.mse_loss(recon, batch).item()
        val_loss /= len(val_loader)

        resample_note = ""
        can_resample = (
            (ep + 1) % resample_interval == 0
            and (ep + 1) <= resample_until
            and dead_pct > 1.0          # only resample when meaningfully dead
        )
        if can_resample:
            n = sae.resample_dead(never_activated, resample_data)
            if n:
                resample_note = f"  [resampled {n}]"

        print(f"  epoch {ep+1:3d}/{epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"dead={dead_pct:.1f}%  "
              f"({time.time()-t0:.1f}s){resample_note}")

    return sae


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True,
                    help="base_*.npy file from collect_activations.py")
    ap.add_argument("--d-sae", type=int, default=16384,
                    help="SAE dictionary size; 8x d_model is a reasonable minimum (default: 16384)")
    ap.add_argument("--k", type=int, default=32,
                    help="TopK sparsity — active features per sample (default: 32)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resample-interval", type=int, default=5,
                    help="Resample dead features every N epochs (default: 5)")
    ap.add_argument("--resample-until", type=int, default=10,
                    help="Stop resampling after this epoch so the second half "
                         "of training converges cleanly (default: 10)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.activations))[0]
        stem = stem.removeprefix("base_")
        sae_dir = "/scratch/inf0/user/nzukowsk/sae"
        os.makedirs(sae_dir, exist_ok=True)
        args.output = f"{sae_dir}/{stem}_d{args.d_sae}_k{args.k}.pt"

    print(f"Loading activations from {args.activations} ...")
    acts = np.load(args.activations, mmap_mode="r")   # memory-mapped: reads from disk per batch
    print(f"  shape: {acts.shape}  dtype: {acts.dtype}")

    d_in = acts.shape[1]
    n    = len(acts)

    # compute mean and std in chunks to avoid loading 34GB into RAM
    print("Computing activation statistics (chunked)...", flush=True)
    chunk = 50_000
    mean = np.zeros(d_in, dtype=np.float64)
    for i in range(0, n, chunk):
        mean += acts[i:i + chunk].sum(axis=0)
    mean /= n

    var = np.zeros(d_in, dtype=np.float64)
    for i in range(0, n, chunk):
        diff = acts[i:i + chunk].astype(np.float64) - mean
        var += (diff ** 2).sum(axis=0)
    std = np.sqrt(var / n)
    mean = mean.astype(np.float32)
    std  = std.astype(np.float32)
    print(f"  done.", flush=True)

    dataset = _ActivationDataset(acts, mean, std)

    val_n   = max(1, int(len(dataset) * 0.1))
    train_n = len(dataset) - val_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(args.seed))

    pin = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin)

    print(f"\nTraining TopK-SAE  d_in={d_in}  d_sae={args.d_sae}  k={args.k}")
    print(f"  train={train_n}  val={val_n}  batch={args.batch_size}  "
          f"epochs={args.epochs}  lr={args.lr}")
    print(f"  resample every {args.resample_interval} epochs "
          f"(epochs 1-{args.resample_until}, only if dead>1%)\n")

    sae = TopKSAE(d_in=d_in, d_sae=args.d_sae, k=args.k)
    sae = train(sae, dataset, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, device=device,
                resample_interval=args.resample_interval,
                resample_until=args.resample_until)

    torch.save({
        "state_dict": sae.cpu().state_dict(),
        "config": {"d_in": d_in, "d_sae": args.d_sae, "k": args.k},
        "norm": {"mean": mean, "std": std},
    }, args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

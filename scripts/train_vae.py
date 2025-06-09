#!/usr/bin/env python3
"""
train_vae.py — 3D convolutional VAE for preprocessed 64³ ShapeNet voxels.

This version normalizes input voxel shapes to exactly 64³ via nearest-neighbor interpolation
and uses safe_collate to avoid non-resizable storage errors.

Usage Example
-------------
python train_vae.py \
    --data ~/projects/cvpr_main/data/ShapeNetVoxel64 \
    --outdir ~/projects/cvpr_main/artifacts/vae \
    --epochs 20 --batch 64 --latent 256 --workers 32
"""

from pathlib import Path
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context
from tqdm import tqdm

# ─────────────────────── safe collate fn ─────────────────────── #
def safe_collate(batch: list) -> torch.Tensor:
    """
    Clone each sample tensor to ensure a fresh, resizable storage before stacking.
    Prevents 'storage not resizable' errors in DataLoader workers.
    """
    return torch.stack([item.clone() for item in batch], dim=0)

# ───────────────────────── Dataset ────────────────────────── #
class VoxelNPZ(Dataset):
    """Load 0/1 occupancy grids (npz or npy), normalize to 64³, return float32 tensor."""
    def __init__(self, root: str, target: int = 64):
        root = Path(root).expanduser()
        self.files = sorted(root.glob('*.npz')) + sorted(root.glob('*.npy'))
        if not self.files:
            raise FileNotFoundError(f"No .npz/.npy files found in {root}")
        self.target = target

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file = self.files[idx]
        if file.suffix == '.npz':
            arr = np.load(file)['occ']
        else:
            arr = np.load(file)
        arr = np.ascontiguousarray(arr.astype(np.float32))
        # if not target shape, interpolate via nearest-neighbor
        if arr.shape != (self.target,)*3:
            v = torch.from_numpy(arr)[None, None]  # shape 1×1×D×H×W
            v = F.interpolate(v, size=(self.target,)*3, mode='nearest')
            arr = v.squeeze().cpu().numpy()
        tensor = torch.tensor(arr, dtype=torch.float32)
        return tensor.unsqueeze(0)  # shape (1,64,64,64)

# ─────────────────────────── Model ─────────────────────────── #
class Encoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64,128,4,2,1),    nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128,256,4,2,1),   nn.BatchNorm3d(256), nn.ReLU(),
            nn.AdaptiveAvgPool3d((4,4,4)),
            nn.Flatten(1)
        )
        self.fc_mu  = nn.Linear(256*4*4*4, z_dim)
        self.fc_log = nn.Linear(256*4*4*4, z_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_log(h)

class Decoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*4*4*4)
        self.net = nn.Sequential(
            nn.Unflatten(1, (256,4,4,4)),
            nn.ConvTranspose3d(256,128,4,2,1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.ConvTranspose3d(128,64,4,2,1),  nn.BatchNorm3d(64),  nn.ReLU(),
            nn.ConvTranspose3d(64,32,4,2,1),   nn.BatchNorm3d(32),  nn.ReLU(),
            nn.ConvTranspose3d(32,1,4,2,1)  # logits output
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        return self.net(h)

class VAE(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.enc(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        logits = self.dec(mu + eps * std)
        return logits, mu, logvar

# ─────────────────────────── Loss ─────────────────────────── #
bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

def loss_fn(logits: torch.Tensor, x: torch.Tensor,
            mu: torch.Tensor, logvar: torch.Tensor,
            beta: float = 1.0):
    bce = bce_loss(logits, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld, bce.detach(), kld.detach()

# ─────────────────────────── Main ─────────────────────────── #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch',  type=int, default=64)
    ap.add_argument('--latent', type=int, default=256)
    ap.add_argument('--lr',     type=float, default=1e-4)
    ap.add_argument('--beta',   type=float, default=1.0, help='KL term weight (β) for loss = BCE + β·KLD')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args() 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # DataLoaders with spawn context
    ctx = get_context('spawn')
    ds = VoxelNPZ(args.data, target=64)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=safe_collate,
        multiprocessing_context=ctx
    )

    prior_dl = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=safe_collate,
        multiprocessing_context=ctx
    )

    # Model, optimizer
    model = VAE(args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging setup
    log_file = outdir / 'log.txt'
    log_file.write_text(json.dumps(vars(args)) + '\n')

    # Training loop
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = total_bce = total_kld = 0.0
        for x in tqdm(dl, desc=f'Epoch {ep}'):
            x = x.to(device)
            logits, mu, logvar = model(x)
            # logits = logits.clamp(-10.0, 10.0)
            loss, bce, kld = loss_fn(logits, x, mu, logvar, beta=args.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); total_bce += bce; total_kld += kld

        msg = f"{ep},{total_loss/len(dl):.4f},{total_bce/len(dl):.4f},{total_kld/len(dl):.4f}"
        print(msg)
        log_file.write_text(log_file.read_text() + msg + '\n')

    # Save model
    torch.save(model.state_dict(), outdir / 'vae_state.pth')

    # Estimate Gaussian prior
    model.eval()
    latents = []
    with torch.no_grad():
        for x in prior_dl:
            x = x.to(device)
            mu, _ = model.enc(x)
            latents.append(mu.cpu())
    all_lat = torch.cat(latents, dim=0)
    z_mean = all_lat.mean(dim=0)
    z_logvar = all_lat.var(dim=0, unbiased=False).log()
    torch.save(torch.cat([z_mean, z_logvar]), outdir / 'z_prior.pt')
    print('z_prior.pt saved')

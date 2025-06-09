#!/usr/bin/env python3
"""
downsample_voxels.py
--------------------
Read 128³ voxel .npy/.npz files in <src-dir>, down-sample them to 64³
(using nearest-neighbor by default), and write the results to <dst-dir>
with the same filenames.

Usage
-----
python downsample_voxels.py \
    --src-dir ~/ShapeNetVoxel128 \
    --dst-dir ~/ShapeNetVoxel64  \
    --target 64                  \
    --mode nearest               # or majority
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def majority_pool(t: torch.Tensor) -> torch.Tensor:
    """2× down-sampling by majority (avg_pool3d + threshold 0.5)."""
    return F.avg_pool3d(t, 2, stride=2).gt_(0.5).float()

def load_voxel(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.load(path)["occ"].astype(np.float32)
    elif path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

def save_voxel(path: Path, arr: np.ndarray) -> None:
    if path.suffix == ".npz":
        np.savez_compressed(path, occ=arr.astype(np.uint8))
    else:  # .npy
        np.save(path, arr.astype(np.uint8))

def downsample(arr: np.ndarray, target: int, mode: str) -> np.ndarray:
    if arr.shape == (target,)*3:
        return arr  # already correct size
    t = torch.from_numpy(np.ascontiguousarray(arr))[None, None]  # 1×1×D×H×W
    if mode == "nearest":
        out = F.interpolate(t, size=(target,)*3, mode="nearest")
    else:  # majority
        # repeatedly apply majority pooling until desired size reached
        while out_shape := out.shape[-1] > target:
            out = majority_pool(t if 'out' not in locals() else out)
    return out.squeeze().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="폴더: 128³ .npy/.npz")
    ap.add_argument("--dst-dir", required=True, help="출력 폴더: 64³")
    ap.add_argument("--target", type=int, default=64, help="목표 해상도")
    ap.add_argument("--mode", choices=["nearest", "majority"], default="nearest",
                    help="down-sampling 방식")
    args = ap.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.glob("*.npy")] + [p for p in src.glob("*.npz")])
    if not files:
        print("⛔  No .npy/.npz files found in", src)
        return

    for f in tqdm(files, desc="Down-sampling"):
        arr = load_voxel(f)
        arr64 = downsample(arr, args.target, args.mode)
        save_voxel(dst / f.name, arr64)

    print(f"✅  {len(files)} files processed → {dst}")

if __name__ == "__main__":
    main()
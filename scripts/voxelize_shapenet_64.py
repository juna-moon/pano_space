#!/usr/bin/env python3
"""
voxelize_shapenet.py
====================
ShapeNet (<cat>/<cat>/<model>/models/model_normalized.obj) ➞ 64³ occupancy grids (.npz) + YAML meta

Key Changes:
* Preferred binvox path: if <model>.surface.binvox exists, load and down-sample to target grid
* Support both OBJ-based mesh voxelization and binvox-based fast path
* New `--mode` option: "nearest" (default) or "majority" down-sampling
* Skip existing outputs for resume support
* Use np.savez (no compression) for maximal speed assuming SSD
* Multiprocessing uses fork on Linux for shared memory reduction

Example
-------
```bash
python voxelize_shapenet.py \
  --input  ~/ShapeNetCore.v2 \
  --out    ~/projects/cvpr_main/data/ShapeNetVoxel64 \
  --grid   64 \
  --mode   majority \
  --workers 16
```
"""

import argparse, os, traceback, yaml, numpy as np
from pathlib import Path
import trimesh as tm
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

# Optional binvox reader if installed
try:
    import binvox_rw
except ImportError:
    binvox_rw = None

# ──────────────────────────────────────────────
def mesh_to_voxel(mesh_f: Path, grid: int = 64, pad: float = 0.01) -> np.ndarray:
    """Convert a mesh OBJ to a 64³ occupancy grid via trimesh."""
    mesh = tm.load(mesh_f, force='mesh')
    if mesh.is_empty:
        return None
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale((1.0 - pad) / max(mesh.extents))
    vox = mesh.voxelized(pitch=1.0 / grid)
    return vox.matrix.astype(np.uint8)

# ──────────────────────────────────────────────
def downsample(arr: np.ndarray, grid: int, mode: str) -> np.ndarray:
    """Down-sample a 3D binary array to (grid³) using nearest or majority."""
    D, H, W = arr.shape
    if (D, H, W) == (grid, grid, grid):
        return arr
    # Ensure divisibility
    factor = D // grid
    if D % grid != 0:
        raise ValueError(f"Cannot down-sample from {D} to {grid}, not divisible")
    if mode == 'nearest':
        return arr[::factor, ::factor, ::factor].astype(np.uint8)
    else:
        # majority pooling: 2×2×2 majority threshold
        out = arr.copy()
        for _ in range(int(np.log2(factor))):
            out = (out.reshape((out.shape[0]//2, 2,
                                out.shape[1]//2, 2,
                                out.shape[2]//2, 2))
                   .sum(axis=(1,3,5)) >= 4).astype(np.uint8)
        return out

# ──────────────────────────────────────────────
def worker(args):
    mesh_f, root, out_root, grid, mode = args
    try:
        # 1) binvox fast-path
        bv = mesh_f.with_suffix('.surface.binvox')
        if bv.exists() and binvox_rw is not None:
            with open(bv, 'rb') as fh:
                vox = binvox_rw.read_as_3d_array(fh).data.astype(np.uint8)
            vox = downsample(vox, grid, mode)
        else:
            vox = mesh_to_voxel(mesh_f, grid)
        if vox is None:
            raise ValueError('empty mesh')

        # 2) output file name
        parts = mesh_f.relative_to(root).parts  # cat / cat / model / ...
        cat, mid = parts[0], parts[2]
        out_f = out_root / f"{cat}_{mid}.npz"

        # 3) skip existing
        if out_f.exists():
            return {'skip': str(out_f)}

        out_f.parent.mkdir(parents=True, exist_ok=True)
        # 4) save with no compression for speed
        np.savez(out_f, occ=vox)
        return {'src': str(mesh_f.relative_to(root)),
                'dst': str(out_f.relative_to(out_root)),
                'grid': grid}
    except Exception as e:
        return {'error': str(e),
                'src': str(mesh_f),
                'trace': traceback.format_exc()}

# ──────────────────────────────────────────────
def collect_objs(root: Path):
    """Collect model_normalized.obj or model.obj within ShapeNet directory tree."""
    objs = []
    for p in root.rglob('models'):
        obj = p / 'model_normalized.obj'
        if not obj.exists():
            obj = p / 'model.obj'
        if obj.exists():
            objs.append(obj)
    return objs

# ──────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True, help='ShapeNet root directory')
    ap.add_argument('--out',    required=True, help='Output voxel directory')
    ap.add_argument('--grid',   type=int, default=64, help='Voxel grid size')
    ap.add_argument('--mode',   choices=['nearest','majority'], default='nearest', help='Down-sample mode')
    ap.add_argument('--workers',type=int, default=16, help='Number of worker processes')
    ap.add_argument('--meta',   default='voxel_meta.yaml', help='YAML metadata filename')
    args = ap.parse_args()

    SHAPE_ROOT = Path(args.input).expanduser()
    OUT_ROOT = Path(args.out).expanduser()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mesh_files = collect_objs(SHAPE_ROOT)
    tasks = [(m, SHAPE_ROOT, OUT_ROOT, args.grid, args.mode) for m in mesh_files]

    # use fork on Linux for faster shared memory
    try:
        set_start_method('fork')
    except RuntimeError:
        pass

    metas, errs = [], []
    with Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc='voxelizing'):
            (errs if 'error' in res else metas).append(res)

    yaml.safe_dump(metas, open(OUT_ROOT/args.meta, 'w'))
    if errs:
        yaml.safe_dump(errs, open(OUT_ROOT/'voxel_errors.yaml','w'))
        print(f"⚠ {len(errs)} errors logged to voxel_errors.yaml")
    skipped = sum(1 for m in metas if 'skip' in m)
    print(f"✓ done — {len(metas)-skipped} new voxels (skipped {skipped})")
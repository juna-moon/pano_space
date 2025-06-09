#!/usr/bin/env python3
"""
voxelize_shapenet.py
====================
ShapeNet( <cat>/<cat>/<model>/models/model_normalized.obj )
→ 64³ occupancy 그리드(.npz) + 메타 YAML.

변경 사항
* binvox(.surface.binvox) 존재 시 직접 로드 → trimesh 변환 skip
* 이미 변환된 .npz 는 건너뜀(skip)  → 중단·재개 지원
* np.savez(압축 0) 로 CPU 사용 ↓, 속도 ↑   (SSD 작업 전제)
* multiprocessing  spawn→fork  기본 변경 (shm 사용↓)
"""

import argparse, os, traceback, yaml, numpy as np
from pathlib import Path
import trimesh as tm
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

# ──────────────────────────────────────────────
def mesh_to_voxel(mesh_f: Path, grid=64, pad=0.01):
    """trimesh → 64³ boolean matrix (uint8)"""
    mesh = tm.load(mesh_f, force='mesh')
    if mesh.is_empty:
        return None
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale((1.0 - pad) / max(mesh.extents))
    vox = mesh.voxelized(pitch=1.0 / grid)
    return vox.matrix.astype(np.uint8)

# ──────────────────────────────────────────────
def worker(args):
    mesh_f, root, out_root, grid = args
    try:
        # ➊ binvox 우선
        bv = mesh_f.with_suffix('.surface.binvox')
        if bv.exists():
            import binvox_rw
            with open(bv, 'rb') as fh:
                vox = binvox_rw.read_as_3d_array(fh).data.astype(np.uint8)
        else:
            vox = mesh_to_voxel(mesh_f, grid)
        if vox is None:
            raise ValueError("empty mesh")

        # ➋ 출력 파일명 (cat_model.npz)
        parts = mesh_f.relative_to(root).parts     # cat / cat / model / …
        cat, mid = parts[0], parts[2]
        out_f = out_root / f"{cat}_{mid}.npz"

        # ➌ 이미 있으면 skip
        if out_f.exists():
            return {"skip": str(out_f)}

        out_f.parent.mkdir(parents=True, exist_ok=True)
        # ➍ 압축 0 (속도↑)
        np.savez(out_f, occ=vox)

        return {"src": str(mesh_f.relative_to(root)),
                "dst": str(out_f.relative_to(out_root)),
                "grid": grid}
    except Exception as e:
        return {"error": str(e),
                "src": str(mesh_f),
                "trace": traceback.format_exc()}

# ──────────────────────────────────────────────
def collect_objs(root: Path):
    """cat/cat/model/models or cat/model/models 모두 탐색"""
    objs = []
    for p in root.rglob("models"):
        obj = p / "model_normalized.obj"
        if not obj.exists():
            obj = p / "model.obj"
        if obj.exists():
            objs.append(obj)
    return objs

# ──────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="ShapeNet root")
    ap.add_argument("--out",    required=True, help="Output voxel dir")
    ap.add_argument("--grid",   type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--meta",   default="voxel_meta.yaml")
    args = ap.parse_args()

    SHAPE_ROOT = Path(args.input).expanduser()
    OUT_ROOT   = Path(args.out).expanduser()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mesh_files = collect_objs(SHAPE_ROOT)
    tasks = [(m, SHAPE_ROOT, OUT_ROOT, args.grid) for m in mesh_files]

    # fork → shared-mem 덜 사용.  (Mac/Windows면 spawn 유지)
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    metas, errs = [], []
    with Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(worker, tasks),
                        total=len(tasks), desc="voxelizing"):
            (errs if "error" in res else metas).append(res)

    yaml.safe_dump(metas, open(OUT_ROOT / args.meta, "w"))
    if errs:
        yaml.safe_dump(errs, open(OUT_ROOT / "voxel_errors.yaml", "w"))
        print(f"⚠ {len(errs)} errors logged to voxel_errors.yaml")
    print(f"✓ done — {len(metas)} voxel files (skipped {len([m for m in metas if 'skip' in m])})")
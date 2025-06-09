#!/usr/bin/env python
"""
convert_core.py
---------------
- src_root 이하 모든 이미지(*.png, *.jpg…)를 재귀 탐색
- 2:1 비율 → 리사이즈만, 그 외 → rectilinear→equirect 투영
- scene/…/rgb.png  →  scene_…_rgb.png  같은 고유 ID로 저장
Usage (모듈 import 전용, 직접 실행 안 함)
"""
import os, math, cv2, json, concurrent.futures
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ────────── rectilinear → equirect ──────────
def rect2equi(img_bgr, out_h):
    h, w = img_bgr.shape[:2]
    out_w = out_h * 2
    jj, ii = np.meshgrid(np.arange(out_w), np.arange(out_h))
    theta = (jj / out_w) * 2 * math.pi - math.pi       # -π ~ π
    phi   = (ii / out_h) * math.pi - math.pi / 2       # -π/2 ~ π/2
    fx = fy = w / math.pi
    cx, cy = w / 2, h / 2
    x = fx * np.cos(phi) * np.sin(theta) + cx
    y = fy * np.sin(phi)              + cy
    map_x, map_y = x.astype(np.float32), y.astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)

# ────────── 메인 함수 ──────────
def convert_dataset(src_root, dst_root, out_wh=(4096, 2048),
                    img_suffixes=(".png", ".jpg", ".jpeg"),
                    qa_csv=None, n_workers=8, base_root=None):
    """Returns (meta_list, error_list)"""
    src_root, dst_root = Path(src_root), Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # QA 매핑 (선택)
    qa_map = {}
    if qa_csv:
        import csv
        with open(qa_csv, newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                qa_map[row['image_id']] = row

    # 작업 리스트
    tasks = []
    for p in src_root.rglob("*"):
        if p.suffix.lower() not in img_suffixes:
            continue
        rel_id = str(p.relative_to(src_root).with_suffix('')).replace(os.sep, '_')
        dst_path = dst_root / f"{rel_id}.png"
        tasks.append((p, src_root, dst_path, out_wh, qa_map.get(rel_id)))

    # 내부 처리 함수
    def _process(args):
        src, root, dst, (W, H), qa_row = args
        try:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if abs(w / h - 2) > 0.05:
                img = rect2equi(img, H)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            base = base_root if base_root else root      # 기준 폴더
            # ── src 경로 ──
            try:
                src_rel = src.relative_to(base)
            except ValueError:
                # 기준 밖이면 안전하게 relpath 사용
                src_rel = Path(os.path.relpath(src, base))

            # ── dst 경로 ──
            try:
                dst_rel = dst.relative_to(base)
            except ValueError:
                dst_rel = Path(os.path.relpath(dst, base))

            return {
                "src": str(src_rel),
                "dst": str(dst_rel),
                "orig_wh": [w, h],
                "qa": qa_row or None,
            }
        except Exception as e:
            return {"error": str(e), "src": str(src)}

    # 병렬 실행
    metas, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        for res in tqdm(ex.map(_process, tasks), total=len(tasks)):
            if "error" in res:
                errors.append(res)
            else:
                metas.append(res)
    return metas, errors
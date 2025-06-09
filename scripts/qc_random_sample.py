#!/usr/bin/env python
"""
qc_random_sample.py
-------------------
지정 폴더에서 무작위 샘플을 뽑아
① PIL·NumPy 로 실제 로드(깨짐 여부 확인)
② R/G/B 평균·표준편차 통계 수집
③ 8×8(기본) 그리드 PNG 로 저장

Usage
$ python qc_random_sample.py --src data/OSR_Bench/pano_4k \
                             --rate 0.01 \
                             --out qc_grid.png \
                             --stats qc_stats.json \
                             --grid 8
"""
import argparse, random, json, math
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

def make_grid(images, grid):
    """images: [PIL] 길이<=grid*grid"""
    w, h = images[0].size
    canvas = Image.new("RGB", (w*grid, h*grid), color=(128,128,128))
    for idx, im in enumerate(images):
        r, c = divmod(idx, grid)
        canvas.paste(im, (c*w, r*h))
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="이미지 폴더(재귀 포함)")
    ap.add_argument("--rate", type=float, default=0.005, help="샘플 비율")
    ap.add_argument("--out", required=True, help="샘플 그리드 PNG")
    ap.add_argument("--stats", required=True, help="JSON 통계 출력")
    ap.add_argument("--grid", type=int, default=8, help="그리드 한 변 개수")
    args = ap.parse_args()

    src = Path(args.src)
    pool = [p for p in src.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg"}]
    k    = max(1, int(len(pool) * args.rate))
    sample_paths = random.sample(pool, k)

    rgb_means, bad = [], []
    thumbs = []

    for p in sample_paths:
        try:
            im = Image.open(p).convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0
            rgb_means.append(arr.reshape(-1,3).mean(axis=0).tolist())
            thumbs.append(ImageOps.fit(im, (256,128), Image.LANCZOS))
        except Exception as e:
            bad.append({"file": str(p), "err": str(e)})

    # 통계 집계
    rgb_arr = np.array(rgb_means)
    stats = {
        "n_total" : len(sample_paths),
        "n_fail"  : len(bad),
        "mean"    : rgb_arr.mean(axis=0).round(4).tolist() if len(rgb_arr) else None,
        "std"     : rgb_arr.std(axis=0).round(4).tolist()  if len(rgb_arr) else None,
        "fails"   : bad,
    }
    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(f"✓ stats saved → {args.stats}")

    # 그리드 저장
    grid_img = make_grid(thumbs, args.grid)
    grid_img.save(args.out)
    print(f"✓ grid saved  → {args.out}")

if __name__ == "__main__":
    main()
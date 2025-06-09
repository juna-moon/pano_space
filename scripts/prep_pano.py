#!/usr/bin/env python
"""
OSR-Bench 파노라마를 4 K equirect PNG로 변환.
 - 원본이 이미 2:1 비율이면 리사이즈만 수행
 - 그 외 비율은 rectilinear(평면) → equirect(구면) 투영 변환
 - QA CSV가 있으면 image_id ↔ 질문·답변 row를 meta JSON에 포함
"""
import argparse, os, json, cv2, math, concurrent.futures, csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ---------- 투영 변환 ----------
import numpy as np
def rectilinear_to_equirect(img_bgr, out_h):
    h, w = img_bgr.shape[:2]
    out_w = out_h * 2
    # θ(경도), φ(위도) 그리드
    jj, ii = np.meshgrid(np.arange(out_w), np.arange(out_h))
    theta = (jj / out_w) * 2 * math.pi - math.pi         # -π ~ π
    phi   = (ii / out_h) * math.pi - math.pi/2            # -π/2 ~ π/2
    # 구면 → 평면 좌표계(간단히 pinhole 모델 사용)
    fx = fy = w / (math.pi)                               # FOV 180°
    cx, cy = w / 2, h / 2
    x = fx * np.cos(phi) * np.sin(theta) + cx
    y = fy * np.sin(phi)          + cy
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)

# ---------- QA CSV 로드 ----------
def load_qa(csv_path):
    if not csv_path: return {}
    qa_map = {}
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            qa_map[row['image_id']] = row
    return qa_map

# ---------- 개별 이미지 처리 ----------
def process_one(args):
    src_path, src_root, dst_path, out_wh, qa_row = args
    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if abs(w / h - 2) > 0.05:      # 2:1 ±5 % 밖이면 투영 변환
            img = rectilinear_to_equirect(img, out_wh[1])
        img = cv2.resize(img, tuple(out_wh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return {
            "src": str(src_path.relative_to(src_root)),
            "dst": str(dst_path.relative_to(src_root)),
            "orig_wh": [w, h],
            "qa": qa_row or None,
        }
    except Exception as e:
        return {"error": str(e), "src": str(src_path)}

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 폴더")
    ap.add_argument("--dst", required=True, help="출력 폴더")
    ap.add_argument("--reso", type=int, nargs=2, default=[4096, 2048])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--qa", help="qa.csv 경로(선택)")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    qa_map = load_qa(args.qa)
    tasks, metas, errors = [], [], []

    for p in src_root.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}: 
            continue
        rel_path = p.relative_to(src_root).with_suffix('')        # scene_xxx/rgb
        rel_id   = str(rel_path).replace(os.sep, '_')             # scene_xxx_rgb
        dst_path = dst_root / f"{rel_id}.png"                     # 고유 파일명
        tasks.append((p, src_root, dst_path, args.reso, qa_map.get(rel_id)))

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for res in tqdm(ex.map(process_one, tasks), total=len(tasks)):
            if "error" in res:
                errors.append(res)
            else:
                metas.append(res)

    # 메타·에러 저장
    Path(args.metadata).write_text(json.dumps(metas, indent=2))
    if errors:
        err_path = args.metadata.replace(".json", "_errors.json")
        Path(err_path).write_text(json.dumps(errors, indent=2))
        print(f"⚠ 변환 실패 {len(errors)}건 → {err_path}")

if __name__ == "__main__":
    main()
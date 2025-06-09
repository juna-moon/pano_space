#!/usr/bin/env python
"""
prep_stanford2d3ds.py (recursive with progress)
---------------------
Stanford2D3Ds 데이터셋 내부의 모든 panos 폴더를
1) 4096×2048 equirectangular 리사이즈
2) 1024×512 슬라이딩 윈도우 (stride=512×256)
출력:
  data/stanford2D3Ds/pano_patches/<scene_id>/*.png
  data/stanford2D3Ds/stanford_meta.json
  data/stanford2D3Ds/stanford_errors.json (실패 시)
"""
import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT     = Path("~/projects/cvpr_main/data/stanford2d3ds").expanduser()
DST      = ROOT / "pano_patches"
META     = ROOT / "stanford_meta.json"
ERR_FILE = ROOT / "stanford_errors.json"

FULL_WH  = (4096, 2048)
PATCH_WH = (1024, 512)
STRIDE   = (PATCH_WH[0]//2, PATCH_WH[1]//2)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def collect_panos():
    # 깊이에 상관없이 모든 panos 디렉터리 찾기
    return [p for p in ROOT.rglob("pano") if p.is_dir()]


def make_patches(img, scene, name, metas, errs):
    pano = img.resize(FULL_WH, Image.LANCZOS)
    out_dir = DST / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = FULL_WH
    pw, ph = PATCH_WH
    sx, sy = STRIDE

    for y in range(0, H - ph + 1, sy):
        for x in range(0, W - pw + 1, sx):
            patch = pano.crop((x, y, x + pw, y + ph))
            fname = f"{name}_x{x:04d}_y{y:04d}.png"
            p = out_dir / fname
            try:
                patch.save(p)
                metas.append({
                    "scene": scene,
                    "orig": name,
                    "patch": str(p.relative_to(ROOT)),
                    "x": x, "y": y
                })
            except Exception as e:
                errs.append({
                    "scene": scene,
                    "orig": name,
                    "patch": fname,
                    "error": str(e)
                })


def main():
    metas, errs = [], []
    panos_dirs = collect_panos()
    print(f"Found {len(panos_dirs)} 'panos' directories to process.")

    # 각 scene 폴더 진행 상황 표시
    for panos_dir in tqdm(panos_dirs, desc="Scenes", unit="scene"):
        scene = panos_dir.parent.name
        img_list = [p for p in panos_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        for img_path in tqdm(img_list, desc=f"Patches {scene}", leave=False, unit="img"):
            name = img_path.stem
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                errs.append({"scene": scene, "orig": name, "error": f"open failed: {e}"})
                continue
            make_patches(img, scene, name, metas, errs)

    # 메타·에러 파일 쓰기
    META.write_text(json.dumps(metas, indent=2))
    if errs:
        ERR_FILE.write_text(json.dumps(errs, indent=2))
        print(f"⚠ 오류 {len(errs)}건 → {ERR_FILE.name}")
        sys.exit(1)

    print(f"✓ Stanford2D3Ds 변환 완료: {len(metas)} patches")


if __name__ == "__main__":
    main()

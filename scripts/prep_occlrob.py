# prep_osr_bench.py
#!/usr/bin/env python
"""
prep_occlrob.py (with progress)
---------------
OcclRobMV 데이터셋 내부의 모든 panos 폴더를
1) 4096×2048 equirectangular 리사이즈
2) 1024×512 슬라이딩 윈도우 (stride=512×256)
출력:
  data/OcclRobMV/pano_patches/<scene_id>/*.png
  data/OcclRobMV/occlrob_meta.json
  data/OcclRobMV/occlrob_errors.json (실패 시)
진행 상황은 씬과 이미지 단위로 tqdm 프로그래스바로 표시됩니다.
"""
import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 경로 설정
ROOT   = Path("~/projects/cvpr_main/data/OcclRobMV").expanduser()
DST    = ROOT / "pano_patches"
META   = ROOT / "occlrob_meta.json"
ERRS   = ROOT / "occlrob_errors.json"

# 리사이즈 및 패치 크기 설정
FULL_WH  = (4096, 2048)  # (W, H)
PATCH_WH = (1024, 512)   # (W, H)
STRIDE   = (PATCH_WH[0]//2, PATCH_WH[1]//2)
IMG_EXTS = (".jpg", ".jpeg", ".png")

# panos 폴더 자동 탐색
def collect_scene_dirs():
    return [d for d in ROOT.iterdir() if (d / "panos").is_dir()]

# 한 씬(scene_dir) 처리 함수
def process_scene(scene_dir, metas, errs):
    scene_id = scene_dir.name
    src_root = scene_dir / "panos"
    # 이미지 파일 리스트
    img_paths = [p for p in src_root.iterdir() if p.suffix.lower() in IMG_EXTS]
    # 이미지별 프로그래스바
    for img_path in tqdm(img_paths, desc=f"Scene {scene_id}", leave=False, unit="img"):
        name = img_path.stem
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            errs.append({"scene":scene_id, "orig":name, "error":f"open failed: {e}"})
            continue

        # 풀 해상도로 리사이즈
        pano = img.resize(FULL_WH, Image.LANCZOS)
        out_dir = DST / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 패치 슬라이딩 윈도우 생성
        W, H = FULL_WH
        pw, ph = PATCH_WH
        sx, sy = STRIDE
        for y in range(0, H - ph + 1, sy):
            for x in range(0, W - pw + 1, sx):
                patch = pano.crop((x, y, x + pw, y + ph))
                fname = f"{name}_x{x:04d}_y{y:04d}.png"
                out_path = out_dir / fname
                try:
                    patch.save(out_path)
                    metas.append({
                        "scene": scene_id,
                        "orig_file": name,
                        "patch_file": str(out_path.relative_to(ROOT)),
                        "x": x, "y": y,
                        "width": pw, "height": ph
                    })
                except Exception as e:
                    errs.append({
                        "scene": scene_id,
                        "orig": name,
                        "patch": fname,
                        "error": str(e)
                    })

# 메인 실행
def main():
    metas, errs = [], []
    scenes = collect_scene_dirs()
    print(f"Found {len(scenes)} scenes to process.")

    # 씬별 프로그래스바
    for scene_dir in tqdm(scenes, desc="Scenes", unit="scene"):
        process_scene(scene_dir, metas, errs)

    # 메타·에러 파일 쓰기
    META.write_text(json.dumps(metas, indent=2))
    if errs:
        ERRS.write_text(json.dumps(errs, indent=2))
        print(f"⚠ Errors: {len(errs)} items → {ERRS.name}")
        sys.exit(1)

    print(f"✓ OcclRobMV 변환 완료: {len(metas)} patches")

if __name__ == "__main__":
    main()

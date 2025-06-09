# prep_muva_ais.py (with progress)
#!/usr/bin/env python
"""
prep_muva_ais.py (recursive with progress)
-----------------
MUVA_AIS 데이터셋의 모든 이미지(재귀) 대상
1) RGB 변환
2) 1024×512 리사이즈
출력:
  data/MUVA_AIS/patches/*.png
  data/MUVA_AIS/muva_meta.json
  data/MUVA_AIS/muva_errors.json (실패 시)
"""
import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT     = Path("~/projects/cvpr_main/data/MUVA_AIS").expanduser()
DST      = ROOT / "patches"
META     = ROOT / "muva_meta.json"
ERR_FILE = ROOT / "muva_errors.json"

TARGET_WH = (1024, 512)
IMG_EXTS  = (".jpg", ".jpeg", ".png")

def main():
    metas, errs = [], []
    img_paths = [p for p in ROOT.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(img_paths)} images in MUVA_AIS.")
    DST.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(img_paths, desc="MUVA_AIS Images", unit="img"):
        rel = img_path.relative_to(ROOT)
        out_name = rel.with_suffix(".png").as_posix().replace("/", "_")
        out_p    = DST/out_name
        try:
            img = Image.open(img_path).convert("RGB")
            patch = img.resize(TARGET_WH, Image.LANCZOS)
            patch.save(out_p)
            metas.append({"orig":str(rel), "patch":str(out_p.relative_to(ROOT))})
        except Exception as e:
            errs.append({"file":str(rel), "error":str(e)})

    META.write_text(json.dumps(metas, indent=2))
    if errs:
        ERR_FILE.write_text(json.dumps(errs, indent=2))
        print(f"⚠ 오류 {len(errs)}건 → {ERR_FILE.name}")
        sys.exit(1)
    print(f"✓ MUVA_AIS 변환 완료: {len(metas)} images")

if __name__ == "__main__":
    main()
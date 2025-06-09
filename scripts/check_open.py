# scripts/check_open.py  (저장 추천)
import os, sys, json, traceback
from pathlib import Path
from PIL import Image
import numpy as np

root = Path(sys.argv[1])           # ex) data/OSR_Bench
bad   = []

def check_one(p: Path):
    try:
        suf = p.suffix.lower()
        if suf in {".jpg", ".jpeg", ".png"}:
            Image.open(p).verify()                 # PIL 무손상 확인
        elif suf in {".npy", ".npz"}:
            np.load(p)
        elif suf == ".binvox":
            # 아주 빠른 헤더만 검사
            with open(p, "rb") as f:
                assert f.read(3) == b'#bi'
        else:
            return
    except Exception as e:
        bad.append({"file": str(p), "err": str(e)})

for fp in root.rglob("*"):
    if fp.is_file():
        check_one(fp)

out = root / "open_check_fail.json"
out.write_text(json.dumps(bad, indent=2))
print(f"검사 완료, 실패 {len(bad)}개 → {out}")
#!/usr/bin/env python3

import json, yaml, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich import print as rprint

def bbox_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    if inter_w == 0 or inter_h == 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter)

def load_metadata(json_path):
    js = json.loads(Path(json_path).read_text())
    # expected: js["objects"] = list of dicts with 'id','bbox','depth'
    df = pd.DataFrame(js["objects"])
    # ensure bbox columns and depth as float
    df['bbox'] = df['bbox'].apply(lambda x: list(map(float, x)))
    df['depth'] = df['depth'].astype(float)
    return df

def tag_visibility(df, delta, partial_thr, full_thr):
    flags = { oid: {"partial": False, "full": False} for oid in df['id'] }
    for i, a in df.iterrows():
        for j, b in df.iterrows():
            if a.id == b.id: continue
            iou = bbox_iou(a.bbox, b.bbox)
            if iou < partial_thr:
                continue
            # depth 누락 처리: NaN -> skip
            if np.isnan(a.depth) or np.isnan(b.depth):
                continue
            if a.depth > b.depth + delta:
                if iou >= full_thr:
                    flags[b.id]["full"] = True
                else:
                    flags[b.id]["partial"] = True
    return flags

def main(args):
    cfg = yaml.safe_load(open(args.cfg)) if args.cfg else {}
    delta      = cfg.get("delta", args.delta)
    partial_thr= cfg.get("partial_thr", args.partial_thr)
    full_thr   = cfg.get("full_thr", args.full_thr)

    pano_paths = sorted(Path(args.pano_root).glob("*.json"))
    if args.samples>0:
        pano_paths = pano_paths[:args.samples]

    results = []
    for jp in tqdm(pano_paths, desc="Scenes"):
        df = load_metadata(jp)
        flags = tag_visibility(df, delta, partial_thr, full_thr)
        results.append({
            "scene": jp.stem,
            "flags": flags
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    rprint(f"[green]✓ saved {len(results)} scenes → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pano-root",   required=True, help="Dir of pano JSONs")
    p.add_argument("--depth-root",  required=False, help="(Future)")
    p.add_argument("--cfg",         default="configs/occl.yaml")
    p.add_argument("--samples",     type=int, default=-1)
    p.add_argument("--delta",       type=float, default=0.15)
    p.add_argument("--partial_thr", type=float, default=0.25)
    p.add_argument("--full_thr",    type=float, default=0.95)
    p.add_argument("--out",         required=True, help="Output JSON path")
    args = p.parse_args()
    main(args)
#!/usr/bin/env python
import json, argparse, re, collections
from pathlib import Path

def norm_text(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

def f1(pred, gold):
    p, g = norm_text(pred).split(), norm_text(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    if not common: return 0.0
    prec = sum(common.values())/len(p)
    rec  = sum(common.values())/len(g)
    return 2*prec*rec/(prec+rec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt",   required=True)
    ap.add_argument("--metrics", default="exact,f1")
    ap.add_argument("--out")
    args = ap.parse_args()

    preds = {d["image_id"]+"_"+d["question"]:d["pred"]
             for d in json.load(open(args.pred))}
    gt    = json.load(open(args.gt))
    exacts, f1s = [], []
    for item in gt:
        key = item["image_id"]+"_"+item["question"]
        if key not in preds: continue
        g, p = item["answer"], preds[key]
        exacts.append(int(norm_text(g)==norm_text(p)))
        f1s.append(f1(p,g))
    res = {"exact":sum(exacts)/len(exacts),
           "f1":   sum(f1s)/len(f1s),
           "n":    len(exacts)}
    print(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res,indent=2))

if __name__ == "__main__":
    main()
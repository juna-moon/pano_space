#!/usr/bin/env python
import argparse, json, torch, tqdm, random
from pathlib import Path
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model= AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")

    rows = (pd.read_csv(args.csv)
              .query("split==@args.split")
              .sample(n=args.num, random_state=args.seed)
              .to_dict("records"))
    preds=[]
    for r in tqdm.tqdm(rows):
        img = Image.open(Path(args.images)/r["image_id"]).convert("RGB")
        prompt = f"Question: {r['question']}\nAnswer:"
        inputs = proc(prompt, img, return_tensors="pt").to(model.device, torch.float16)
        out = model.generate(**inputs, max_new_tokens=32)
        txt = proc.decode(out[0], skip_special_tokens=True).strip()
        preds.append({"image_id":r["image_id"],"question":r["question"],"pred":txt})
    Path(args.out).write_text(json.dumps(preds,indent=2))
    print("✓ LLaVA preds saved", len(preds))

if __name__ == "__main__":
    main()
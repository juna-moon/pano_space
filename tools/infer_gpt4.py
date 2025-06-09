#!/usr/bin/env python
import argparse, base64, io, json, os, time, random, pandas as pd
from pathlib import Path
from PIL import Image
import openai, tqdm

#— 추가: HTTP 프록시나 alternate API base URL 지원
#   예) export OPENAI_API_BASE="https://api.openrouter.ai/v1"
#       export OPENAI_API_KEY="YOUR_OPENROUTER_TOKEN"
# 또는 커맨드라인 인자로 --api_base, --api_key 전달 가능

def load_resize(img_f, max_wh=1024):
    im = Image.open(img_f).convert("RGB")
    w, h = im.size
    scale = max_wh / max(w, h)
    if scale < 1.0:
        im = im.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO(); im.save(buf, format="JPEG"); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def ask_gpt4(img_b64, question):
    # 신규 v1.0+ 인터페이스: OpenAI() client 객체 사용
    client = openai.OpenAI()
    payload = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text",      "text": question}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # 필요에 따라 다른 모델로 변경 가능
        messages=[{"role": "user", "content": payload}],
        max_tokens=32,
        temperature=0.0
    )
    # 응답 메시지 반환
    return resp.choices[0].message.content.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", help="API Base URL (e.g. https://api.openrouter.ai/v1)")
    ap.add_argument("--api_key",  help="API Key or Token for the chosen endpoint")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    #— OpenAI 설정: 커맨드라인 > env var > default
    if args.api_base:
        openai.api_base = args.api_base
    elif os.getenv("OPENAI_API_BASE"):
        openai.api_base = os.getenv("OPENAI_API_BASE")

    if args.api_key:
        openai.api_key = args.api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    random.seed(args.seed)
    # C 엔진에서 오류 날 경우 Python 엔진으로 재시도하며, bad line은 건너뜁니다.
    try:
        df = pd.read_csv(args.csv)
    except pd.errors.ParserError:
        print("⚠ ParserError 발생: Python 엔진으로 재시도하며 bad lines를 건너뜁니다.")
        df = pd.read_csv(
            args.csv,
            engine="python",
            on_bad_lines="skip",    # 문제 있는 행은 건너뜁니다 (Warn 모드도 가능)
            sep=",",
            quotechar='"'
        )
    print("=== DataFrame columns ===")
    print(df.columns.tolist())

    # CSV에 split 컬럼이 없으면 전체 데이터에서 샘플링
    if "split" in df.columns:
        rows = df[df["split"] == args.split] \
                .sample(n=args.num, random_state=args.seed)
    else:
        print("⚠ Warning: 'split' column not found. Sampling from entire DataFrame.")
        rows = df.sample(n=args.num, random_state=args.seed)

    results = []
    for i, row in enumerate(tqdm.tqdm(rows.to_dict("records"))):
        img_f = Path(args.images)/row["image_id"]
        img_b64 = load_resize(img_f)
        ans = ask_gpt4(img_b64, row["question"])
        results.append({"image_id":row["image_id"],
                        "question":row["question"],
                        "pred":ans})
        if (i+1) % args.batch == 0:      # 간단 딜레이(속도 제한)
            time.sleep(1.5)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"✓ saved {len(results)} preds → {args.out}")

if __name__ == "__main__":
    main()
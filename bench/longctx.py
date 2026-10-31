#!/usr/bin/env python3
"""Salted needle sweep: recall, prefill speed and TTFT by prompt length.

Unique filler per length and depth (a random nonce seeds the word list), so no
request can hit the prefix cache. A 6-character code sits at the requested
depth; PASS when the reply's content or reasoning contains it.

  python3 bench/longctx.py --base-url http://192.0.2.11:8000 --model minimax-m3.1-flash
  ... --lengths 8k,32k,64k,128k,256k --depths 10,50 --out runs/longctx

Lengths above the server's max_model_len are skipped, not clamped silently.
The reported prompt_tokens is the engine's own count.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

WORDS = (
    "almanac beacon copper delta ember fathom gable harbour isotope juniper kelp lantern meadow "
    "nectar orchard pollen quarry ridge sable tundra umber vessel willow xenon yarrow zephyr"
).split()


def parse_size(text: str) -> int:
    text = text.strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    if text.endswith("m"):
        return int(float(text[:-1]) * 1024 * 1024)
    return int(text)


def models(base: str, api_key: str) -> dict:
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    req = urllib.request.Request(base.rstrip("/") + "/v1/models", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())
    return (payload.get("data") or [{}])[0]


def stream(url: str, body: dict, api_key: str, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    started = time.time()
    ttft = None
    last = None
    text = []
    reasoning = []
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                reason = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    text.append(piece)
                if reason:
                    reasoning.append(reason)
                if piece or reason:
                    now = time.time()
                    ttft = ttft if ttft is not None else now - started
                    last = now
    return {
        "wall_s": time.time() - started,
        "ttft_s": ttft,
        "decode_s": (last - started) if last else None,
        "usage": usage,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
    }


def filler(words: int, nonce: str, depth: int) -> tuple[str, str]:
    rng = random.Random(nonce + ":" + str(depth))
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    head_n = max(1, words * depth // 100)
    body = " ".join(rng.choice(WORDS) for _ in range(head_n))
    tail = " ".join(rng.choice(WORDS) for _ in range(max(1, words - head_n)))
    return f"{nonce} {body}\n\nThe access code is {code}.\n\n{tail}", code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--lengths", default="8k,32k,64k,128k,256k")
    ap.add_argument("--depths", default="10,50")
    ap.add_argument("--word-ratio", type=float, default=1.3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", help="JSON output path")
    args = ap.parse_args()

    info = models(args.base_url, args.api_key)
    limit = info.get("max_model_len")
    print(f"model {info.get('id')} max_model_len {limit}", flush=True)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rows = []
    for target in [parse_size(x) for x in args.lengths.split(",") if x.strip()]:
        if limit and target > int(limit):
            print(f"{target}: skipped (> max_model_len {limit})")
            continue
        for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
            nonce = uuid.uuid4().hex[:10]
            words = max(64, int(target / args.word_ratio))
            text, code = filler(words, nonce, depth)
            prompt = text + "\n\nWhat is the access code? Reply with the code only."
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }
            try:
                row = stream(url, body, args.api_key, args.timeout)
                usage = row["usage"] or {}
                prompt_tokens = usage.get("prompt_tokens")
                ttft = row["ttft_s"]
                prefill = (prompt_tokens / ttft) if prompt_tokens and ttft else None
                completed = usage.get("completion_tokens")
                decode = ((completed - 1) / (row["decode_s"] - ttft)) if completed and ttft and row["decode_s"] and row["decode_s"] > ttft else None
                answer = row["text"] + row["reasoning"]
                ok = code in answer
                rows.append(
                    {
                        "target": target, "depth_pct": depth, "prompt_tokens": prompt_tokens,
                        "ttft_s": round(ttft, 3) if ttft else None,
                        "prefill_tok_s": round(prefill, 1) if prefill else None,
                        "decode_tok_s": round(decode, 1) if decode else None,
                        "ok": ok, "code": code,
                    }
                )
                print(
                    f"  {target:>8} @ {depth:>2}%  prompt {prompt_tokens}  TTFT {rows[-1]['ttft_s']}s  "
                    f"prefill {rows[-1]['prefill_tok_s']} tok/s  {'PASS' if ok else 'FAIL'}",
                    flush=True,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                rows.append({"target": target, "depth_pct": depth, "ok": False, "error": str(exc)})
                print(f"  {target} @ {depth}%: FAILED {exc}", file=sys.stderr)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "max_model_len": limit, "rows": rows}, f, indent=1)
        print(f"wrote {args.out}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

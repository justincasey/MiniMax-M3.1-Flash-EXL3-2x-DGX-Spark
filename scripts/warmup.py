#!/usr/bin/env python3
"""Warm-up requests for the MiniMax-M3.1 TP2 serve.

Two requests, before the READY line:
  1. a short chat (absorbs first-request allocator/JIT transients)
  2. an ~2.4K-token prompt (first prefill-sized batch under the guard). An 8.4K prompt was
     measured: it adds ~5 s to ready_s, runs mixed with the smoke requests,
     and did not change the first 8K prefill afterwards.

Every request's wall time, usage and finish_reason is appended to
<run>/warmup.jsonl. A non-200 or a timeout fails the start.

  python3 scripts/warmup.py --base-url http://host:8000 --model minimax-m3.1-flash --run runs/x
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

FILLER = (
    "A sparse-attention mixture-of-experts decoder routes each token to a small set of experts. "
    "The scheduler batches prefills and decodes, the cache holds one block per 128 tokens, and the "
    "speculative drafter verifies several tokens per step against the target distribution. "
)


def post(url: str, body: dict, api_key: str, timeout: int = 900) -> tuple[dict, float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    nonce = uuid.uuid4().hex[:12]
    long_prompt = (
        "Read the following notes, then reply with exactly the word READY.\n\n"
        + FILLER * 40
        + f"\n\nNonce: {nonce}. Reply READY."
    )
    cases = [
        ("short", [{"role": "user", "content": "Reply with the single word READY."}], 32),
        ("long", [{"role": "user", "content": long_prompt}], 32),
    ]
    os.makedirs(args.run, exist_ok=True)
    out = os.path.join(args.run, "warmup.jsonl")
    failures = 0
    with open(out, "a") as log:
        for name, messages, max_tokens in cases:
            body = {
                "model": args.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            try:
                payload, wall = post(url, body, args.api_key)
                usage = payload.get("usage") or {}
                row = {
                    "case": name,
                    "wall_s": round(wall, 3),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
                    "ok": True,
                }
                print(f"warmup {name}: {wall:.1f}s, prompt {row['prompt_tokens']} tokens", flush=True)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                failures += 1
                row = {"case": name, "ok": False, "error": str(exc)}
                print(f"warmup {name}: FAILED: {exc}", file=sys.stderr)
            log.write(json.dumps(row) + "\n")
            log.flush()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fixed 50-question exact-answer gate (40 seeded multiplications + 10 capitals).

Temperature 0, adaptive thinking (the MiniMax template decides), max_tokens
1024 so the reasoning fits. The answer is taken from message.content, never
from reasoning; a reply with no visible answer counts wrong.

  python3 bench/quality_gate.py http://192.0.2.11:8000 --model minimax-m3.1-flash
  ... --out answers.json --min 45     # exit 1 below --min

The first run establishes the baseline; gate later runs against it.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import urllib.request

CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ("Brazil", "Brasília"), ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Egypt", "Cairo"),
    ("Peru", "Lima"), ("Vietnam", "Hanoi"),
]


def questions() -> list[tuple[str, str, str]]:
    rng = random.Random(1017)
    items = []
    for _ in range(40):
        a, b = rng.randint(12, 999), rng.randint(12, 99)
        items.append((f"multiply {a}x{b}", f"What is {a}*{b}? Reply with the integer only.", str(a * b)))
    for country, capital in CAPITALS:
        items.append((f"capital {country}", f"What is the capital of {country}? Reply with the city name only.", capital))
    return items


def ask(url: str, model: str, prompt: str, api_key: str, effort: str | None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def grade(kind: str, want: str, content: str) -> bool:
    if not content:
        return False
    if kind.startswith("multiply"):
        numbers = re.findall(r"-?\d[\d,]*", content)
        if not numbers:
            return False
        return numbers[-1].replace(",", "") == want
    return want.lower() in content.lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_url")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--out")
    ap.add_argument("--min", type=int, default=0, help="fail below this many correct (of 50)")
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    items = questions()

    def run(item):
        kind, prompt, want = item
        try:
            payload = ask(url, args.model, prompt, args.api_key, args.effort)
            message = (payload.get("choices") or [{}])[0].get("message") or {}
            content = (message.get("content") or "").strip()
            usage = payload.get("usage") or {}
            return {
                "kind": kind, "prompt": prompt, "want": want, "content": content,
                "ok": grade(kind, want, content),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"kind": kind, "prompt": prompt, "want": want, "content": "", "ok": False, "error": str(exc)}

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(run, items))

    correct = sum(1 for r in rows if r["ok"])
    print(f"accuracy {correct}/{len(rows)}")
    for r in rows:
        if not r["ok"]:
            print(f"  miss {r['kind']}: want {r['want']!r}, got {r['content'][:120]!r}"
                  + (f" ({r['error']})" if r.get("error") else ""), file=sys.stderr)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"accuracy": correct, "total": len(rows), "min": args.min, "rows": rows}, f, indent=1)
    return 0 if correct >= args.min else 1


if __name__ == "__main__":
    raise SystemExit(main())

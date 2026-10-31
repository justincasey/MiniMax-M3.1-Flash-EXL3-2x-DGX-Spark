#!/usr/bin/env python3
"""Pre-flight capability probe: run before trusting any benchmark number.

Checks, in order:
  1. /v1/models answers and reports the served id and max_model_len
  2. /tokenize exists and counts a known string
  3. a chat reply is coherent (math) and reasoning arrives separately
  4. a tool call parses to a function name + JSON arguments
  5. /metrics exports DSpark spec-decode counters after one drafted request
  6. reasoning_effort (if supported by the template) does not 400

  python3 bench/probe.py --base-url http://192.0.2.11:8000 --model minimax-m3.1-flash

Exit 1 on any hard failure; a missing metrics family is a warning (the server
may disable metrics), but a missing spec counter with DSPARK=1 is a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def req(url: str, body: dict | None, api_key: str, timeout: int = 600) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--dspark", type=int, default=1)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    fails = 0

    def ok(name: str, detail: str = ""):
        print(f"PASS {name}" + (f": {detail}" if detail else ""))

    def bad(name: str, detail: str):
        nonlocal fails
        fails += 1
        print(f"FAIL {name}: {detail}")

    try:
        listing = req(base + "/v1/models", None, args.api_key)
        ids = [row.get("id") for row in listing.get("data") or []]
        maxlen = (listing.get("data") or [{}])[0].get("max_model_len")
        ok("models", f"{ids} max_model_len={maxlen}")
        if args.model not in ids:
            bad("model id", f"{args.model!r} not served; got {ids}")
    except Exception as exc:  # noqa: BLE001
        bad("models", str(exc))
        print("probe cannot continue without /v1/models", file=sys.stderr)
        return 1

    try:
        tokens = req(base + "/tokenize", {"model": args.model, "prompt": "hello world", "add_special_tokens": False}, args.api_key)
        count = tokens.get("count", len(tokens.get("tokens") or []))
        ok("tokenize", f"{count} tokens")
    except Exception as exc:  # noqa: BLE001
        bad("tokenize", str(exc))

    try:
        chat = req(base + "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is 17*23? Reply with the integer only."}],
            "max_tokens": 1024,
            "temperature": 0,
        }, args.api_key)
        message = (chat.get("choices") or [{}])[0].get("message") or {}
        content = (message.get("content") or "")
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        usage = chat.get("usage") or {}
        if "391" in content:
            ok("chat", "17*23 -> 391")
        else:
            bad("chat", f"expected 391, got {content[:120]!r}")
        if reasoning:
            ok("reasoning separation", f"{len(reasoning)} chars")
        else:
            print("WARN reasoning separation: no reasoning/reasoning_content field (adaptive mode may skip thinking)")
        if usage.get("completion_tokens"):
            ok("usage", f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
        else:
            bad("usage", "no completion_tokens in the response")
    except Exception as exc:  # noqa: BLE001
        bad("chat", str(exc))

    try:
        tool = req(base + "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
            "tools": [TOOL],
            "tool_choice": "auto",
            "max_tokens": 1024,
            "temperature": 0,
        }, args.api_key)
        calls = ((tool.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
        if calls and (calls[0].get("function") or {}).get("name") == "get_weather":
            json.loads((calls[0]["function"].get("arguments") or "{}"))
            ok("tool call", "get_weather with JSON arguments")
        else:
            bad("tool call", f"no get_weather call: {calls}")
    except Exception as exc:  # noqa: BLE001
        bad("tool call", str(exc))

    try:
        text = urllib.request.urlopen(
            urllib.request.Request(base + "/metrics", headers={"Authorization": "Bearer " + args.api_key} if args.api_key else {}),
            timeout=30,
        ).read().decode()
        names = [line.split("{")[0] for line in text.splitlines() if line and not line.startswith("#")]
        spec = sorted({n for n in names if "spec_decode" in n})
        if spec:
            ok("metrics spec_decode", ", ".join(spec[:4]))
        elif args.dspark:
            bad("metrics spec_decode", "no spec_decode counters exported; DSpark is on")
        else:
            print("WARN metrics: no spec_decode counters (DSPARK=0)")
        if any("kv_cache_usage" in n for n in names):
            ok("metrics kv_cache_usage")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN metrics: {exc}")

    try:
        effort = req(base + "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 64,
            "temperature": 0,
            "reasoning_effort": "low",
        }, args.api_key)
        ok("reasoning_effort", "accepted")
    except urllib.error.HTTPError as exc:
        print(f"WARN reasoning_effort: HTTP {exc.code} (the template may not accept it; benches must not pass it)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN reasoning_effort: {exc}")

    print(f"probe: {'OK' if fails == 0 else str(fails) + ' failure(s)'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

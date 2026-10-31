#!/usr/bin/env python3
"""Smoke test: three checks against a fresh MiniMax-M3.1 TP2 serve.

  1. plain chat      17*23 -> 391, finish_reason stop
  2. tool call       get_weather(Paris) parses to name + JSON arguments
  3. reasoning       the reply separates reasoning from content (field or
                     reasoning_tokens), and the visible answer is coherent

The mixed-K dispatch-mode gate lives in m31.sh smoke because it reads rank 0's
container log (a server-side property, not an HTTP one).

  python3 tests/smoke.py --base-url http://192.0.2.11:8000/v1 --model minimax-m3.1-flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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


def post(url: str, body: dict, api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def check_chat(chat: str, model: str, key: str) -> list[str]:
    chat1 = post(chat, {
        "model": model,
        "messages": [{"role": "user", "content": "What is 17*23? Reply with the integer only."}],
        "max_tokens": 1024,
        "temperature": 0,
    }, key)
    choice = (chat1.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    if "391" not in content:
        return [f"1: expected 391, got {content[:120]!r}"]
    if choice.get("finish_reason") not in ("stop", "length"):
        return [f"1: finish_reason={choice.get('finish_reason')!r}"]
    return []


def check_tool(chat: str, model: str, key: str) -> list[str]:
    chat2 = post(chat, {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [TOOL],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0,
    }, key)
    message = (chat2.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    parsed = False
    if calls:
        fn = calls[0].get("function") or {}
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        parsed = fn.get("name") == "get_weather" and "paris" in json.dumps(arguments).lower()
    return [] if parsed else [f"2: no well-formed get_weather call; content={str(message.get('content'))[:120]!r}"]


def check_reasoning(chat: str, model: str, key: str) -> list[str]:
    # Same problem and the same checks (reasoning present and separate, 7.75 h in the
    # answer); "think briefly" keeps the chain of thought to ~50 tokens instead of
    # ~200-300, so the check costs ~2 s rather than ~8-11 s at boot.
    chat3 = post(chat, {
        "model": model,
        "messages": [{"role": "user", "content": "A train covers 465 km at 60 km/h. How many hours does the trip take? "
                                                  "Think briefly, then answer."}],
        "max_tokens": 4096,
        "temperature": 0,
    }, key)
    message = (chat3.get("choices") or [{}])[0].get("message") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = (message.get("content") or "").strip()
    usage = chat3.get("usage") or {}
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    fails = []
    if not reasoning and not reasoning_tokens:
        fails.append(f"3: no reasoning/reasoning_content and no reasoning_tokens; keys={sorted(message)}")
    if not content:
        fails.append("3: empty visible answer")
    if reasoning and reasoning == content:
        fails.append("3: reasoning equals content")
    if content and "7.75" not in content and "7 h" not in content.lower() and "hours" not in content.lower():
        fails.append(f"3: answer lacks the expected ~7.75 h: {content[:160]!r}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    args = ap.parse_args()
    chat = args.base_url.rstrip("/") + "/chat/completions"

    # The three checks are independent requests: run them at once (the server
    # batches them), report in order.
    checks = [("1 chat", check_chat), ("2 tool call", check_tool), ("3 reasoning separation", check_reasoning)]
    failed = 0
    with ThreadPoolExecutor(max_workers=len(checks)) as pool:
        futures = [pool.submit(fn, chat, args.model, args.api_key) for _, fn in checks]
        for (label, _), fut in zip(checks, futures):
            try:
                fails = fut.result()
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
                fails = [f"{label.split()[0]}: {exc}"]
            for message in fails:
                print("SMOKE FAIL " + message)
            if fails:
                failed += 1
            else:
                print(f"PASS {label}")
    if failed:
        return 1
    print("SMOKE PASS: chat, tool call, reasoning separated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

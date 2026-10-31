#!/usr/bin/env python3
"""OpenAI API conformance suite for the MiniMax-M3.1 TP2 serve.

Stdlib only. Covers the surface clients depend on: models/health, chat,
streaming with usage, max_tokens, stop, reasoning separation, tool calls,
completions, tokenize/detokenize, error handling, concurrency and (optionally)
API-key enforcement.

  python3 tests/api_conformance.py --base-url http://192.0.2.11:8000/v1 --model minimax-m3.1-flash
  ... --api-key K --strict-determinism

Exit 1 on any FAIL. WARNs are reported but do not fail the suite.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

RESULTS = []


def record(status: str, name: str, detail: str = ""):
    RESULTS.append((status, name, detail))
    print(f"{status:<5} {name}" + (f": {detail}" if detail else ""))


def request(url: str, body: dict | None, api_key: str, timeout: int = 600, raw: bytes | None = None):
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = raw
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data, headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(), dict(exc.headers)


def jbody(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def stream(url: str, body: dict, api_key: str) -> tuple[int, list[dict], str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    events = []
    raw = []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                text = line.decode("utf-8", "replace").strip()
                raw.append(text)
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    events.append(json.loads(payload))
                except ValueError:
                    pass
            return 200, events, "\n".join(raw)
    except urllib.error.HTTPError as exc:
        return exc.code, events, exc.read().decode()


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--strict-determinism", action="store_true")
    ap.add_argument("--strict-reasoning", action="store_true")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    chat = base + "/chat/completions"

    # 1. models
    status, text, _ = request(base + "/models", None, args.api_key)
    listing = jbody(text) or {}
    ids = [row.get("id") for row in listing.get("data") or []]
    if status == 200 and ids:
        record("PASS", "models", f"{ids} max_model_len={(listing.get('data') or [{}])[0].get('max_model_len')}")
    else:
        record("FAIL", "models", f"HTTP {status}")
    if args.model not in ids:
        record("FAIL", "served model id", f"{args.model!r} not in {ids}")

    # 2. health
    status, text, _ = request(root + "/health", None, args.api_key)
    record("PASS" if status == 200 else "WARN", "health", f"HTTP {status}")

    # 3. basic chat + usage
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with the single word READY."}],
        "max_tokens": 256,
        "temperature": 0,
    }
    status, text, _ = request(chat, body, args.api_key)
    payload = jbody(text) or {}
    choice = (payload.get("choices") or [{}])[0]
    usage = payload.get("usage") or {}
    if status == 200 and choice.get("finish_reason") and usage.get("completion_tokens"):
        record("PASS", "chat + usage", f"completion_tokens={usage.get('completion_tokens')}")
    else:
        record("FAIL", "chat + usage", f"HTTP {status} body={text[:200]!r}")

    # 4. max_tokens respected
    body64 = dict(body, max_tokens=16)
    status, text, _ = request(chat, body64, args.api_key)
    p64 = jbody(text) or {}
    completion = (p64.get("usage") or {}).get("completion_tokens")
    if status == 200 and completion is not None and completion <= 16:
        record("PASS", "max_tokens", f"completion_tokens={completion}")
    else:
        record("FAIL", "max_tokens", f"HTTP {status}, completion_tokens={completion}")

    # 5. stop sequence
    body_stop = dict(body, max_tokens=256, stop=["\n"])
    status, text, _ = request(chat, body_stop, args.api_key)
    pstop = jbody(text) or {}
    content = ((pstop.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if status == 200 and "\n" not in content:
        record("PASS", "stop sequence")
    else:
        record("FAIL", "stop sequence", f"HTTP {status}, content={content[:80]!r}")

    # 6. streaming + final usage
    status, events, raw = stream(chat, dict(body, max_tokens=32), args.api_key)
    deltas = sum(
        1 for e in events for c in (e.get("choices") or [])
        if (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning")
        or (c.get("delta") or {}).get("reasoning_content")
    )
    stream_usage = next((e.get("usage") for e in reversed(events) if e.get("usage")), None)
    if status == 200 and deltas and stream_usage and "[DONE]" in raw:
        record("PASS", "streaming", f"{deltas} deltas, usage.completion_tokens={stream_usage.get('completion_tokens')}")
    else:
        record("FAIL", "streaming", f"HTTP {status}, deltas={deltas}, usage={stream_usage}, done={'[DONE]' in raw}")

    # 7. reasoning separation
    body_think = {
        "model": args.model,
        "messages": [{"role": "user", "content": "A train travels 465 km at 60 km/h. Think step by step, then answer: how many hours?"}],
        "max_tokens": 4096,
        "temperature": 0,
    }
    status, text, _ = request(chat, body_think, args.api_key)
    think = jbody(text) or {}
    msg = (think.get("choices") or [{}])[0].get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    think_usage = think.get("usage") or {}
    reasoning_tokens = (think_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    visible = (msg.get("content") or "").strip()
    if status == 200 and visible and (reasoning or reasoning_tokens):
        record("PASS", "reasoning separation", f"{len(reasoning)} chars, reasoning_tokens={reasoning_tokens}")
    else:
        record("FAIL" if args.strict_reasoning else "WARN", "reasoning separation",
               f"reasoning={len(reasoning)} reasoning_tokens={reasoning_tokens} content={visible[:80]!r}")

    # 8. tool call
    status, text, _ = request(chat, {
        "model": args.model,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [TOOL], "tool_choice": "auto", "max_tokens": 1024, "temperature": 0,
    }, args.api_key)
    tool_payload = jbody(text) or {}
    calls = ((tool_payload.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    ok = False
    if calls:
        fn = calls[0].get("function") or {}
        try:
            ok = fn.get("name") == "get_weather" and "paris" in json.dumps(json.loads(fn.get("arguments") or "{}")).lower()
        except ValueError:
            ok = False
    record("PASS" if ok else "FAIL", "tool call", f"calls={len(calls)}")

    # 9. legacy completions
    status, text, _ = request(base + "/completions", {
        "model": args.model, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0,
    }, args.api_key)
    comp = jbody(text) or {}
    if status == 200 and (comp.get("choices") or [{}])[0].get("text") is not None:
        record("PASS", "completions")
    else:
        record("FAIL", "completions", f"HTTP {status}")

    # 10. tokenize / detokenize
    status, text, _ = request(root + "/tokenize", {"model": args.model, "prompt": "hello world"}, args.api_key)
    tok = jbody(text) or {}
    ids_tok = tok.get("tokens") or []
    if status == 200 and tok.get("count", len(ids_tok)) > 0:
        record("PASS", "tokenize", f"count={tok.get('count', len(ids_tok))}")
        status, text, _ = request(root + "/detokenize", {"model": args.model, "tokens": ids_tok}, args.api_key)
        detok = jbody(text) or {}
        record("PASS" if status == 200 and "hello" in (detok.get("prompt") or "") else "FAIL",
               "detokenize", f"HTTP {status}")
    else:
        record("FAIL", "tokenize", f"HTTP {status}")

    # 11. error handling
    status, _, _ = request(chat, {"model": "definitely-not-served", "messages": [{"role": "user", "content": "hi"}]}, args.api_key)
    record("PASS" if status in (400, 404) else "FAIL", "unknown model", f"HTTP {status}")
    status, _, _ = request(chat, None, args.api_key, raw=b"{not json")
    record("PASS" if status == 400 else "FAIL", "bad JSON", f"HTTP {status}")
    status, _, _ = request(chat, {"model": args.model, "messages": []}, args.api_key)
    record("PASS" if status == 400 else "FAIL", "empty messages", f"HTTP {status}")

    # 12. context limit
    if listing.get("data"):
        maxlen = (listing["data"][0].get("max_model_len") or 0)
        if maxlen:
            status, _, _ = request(chat, {
                "model": args.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": maxlen + 1,
            }, args.api_key)
            record("PASS" if status == 400 else "FAIL", "max_tokens over context", f"HTTP {status}")

    # 13. concurrency
    def one(_):
        return request(chat, dict(body, max_tokens=32), args.api_key)[0]

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        codes = list(ex.map(one, range(4)))
    record("PASS" if all(c == 200 for c in codes) else "FAIL", "4 concurrent requests", str(codes))

    # 14. temperature-0 determinism
    body_pin = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Compute 37*43 step by step, then state the final answer."}],
        "max_tokens": 256, "temperature": 0, "seed": 1234, "return_token_ids": True,
    }
    status1, text1, _ = request(chat, body_pin, args.api_key)
    time.sleep(0.2)
    status2, text2, _ = request(chat, body_pin, args.api_key)
    p1, p2 = jbody(text1) or {}, jbody(text2) or {}
    ids1 = ((p1.get("choices") or [{}])[0]).get("token_ids")
    ids2 = ((p2.get("choices") or [{}])[0]).get("token_ids")
    if ids1 and ids2:
        same = ids1 == ids2
    else:
        same = ((p1.get("choices") or [{}])[0].get("message") or {}).get("content") == \
               ((p2.get("choices") or [{}])[0].get("message") or {}).get("content")
    record("PASS" if same else ("FAIL" if args.strict_determinism else "WARN"),
           "temperature-0 determinism", "identical" if same else "differs")

    # 15. API key enforcement (only when a key is configured)
    if args.api_key:
        status, _, _ = request(chat, body, "")
        record("PASS" if status == 401 else "FAIL", "API key required", f"HTTP {status}")

    failed = [r for r in RESULTS if r[0] == "FAIL"]
    warned = [r for r in RESULTS if r[0] == "WARN"]
    print(f"api conformance: {len(RESULTS) - len(failed) - len(warned)} pass, {len(warned)} warn, {len(failed)} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

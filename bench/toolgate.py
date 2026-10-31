#!/usr/bin/env python3
"""Deterministic tool-calling gate for an OpenAI-compatible endpoint.

27 scripted checks (24 single-call scenarios plus multi-turn follow-ups)
exercising the MiniMax XML tool parser: required and optional arguments,
enums, arrays, nested objects, parallel calls, type coercion, no-tool
questions, a tool-result follow-up and one error recovery.

  python3 bench/toolgate.py --base-url http://192.0.2.11:8000/v1 --model minimax-m3.1-flash
  ... --min-pass 20 --out runs/toolgate.json

Grading is on the parsed tool_calls (name + JSON arguments), never on raw text.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "cc": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "qty": {"type": "integer"},
                            },
                            "required": ["sku", "qty"],
                        },
                    },
                    "expedited": {"type": "boolean"},
                },
                "required": ["customer_id", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_lang": {"type": "string", "enum": ["es", "fr", "de"]},
                },
                "required": ["text", "target_lang"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "readonly": {"type": "boolean"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def _type_ok(value, schema) -> bool:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        for key in schema.get("required", []):
            if key not in value:
                return False
        props = schema.get("properties", {})
        return all(_type_ok(v, props[k]) for k, v in value.items() if k in props)
    if kind == "array":
        return isinstance(value, list) and all(_type_ok(v, schema.get("items", {})) for v in value)
    if kind == "string":
        ok = isinstance(value, str)
        return ok and (not schema.get("enum") or value in schema["enum"])
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def _calls(response: dict) -> list[dict]:
    message = (response.get("choices") or [{}])[0].get("message") or {}
    return message.get("tool_calls") or []


def _content(response: dict) -> str:
    message = (response.get("choices") or [{}])[0].get("message") or {}
    return (message.get("content") or "").strip()


def _judge_call(call: dict, want: dict) -> tuple[bool, str]:
    fn = call.get("function") or {}
    if fn.get("name") != want["name"]:
        return False, f"called {fn.get('name')!r}, want {want['name']!r}"
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return False, f"arguments are not JSON: {fn.get('arguments')[:80]!r}"
    schema = TOOL_BY_NAME[want["name"]]["parameters"]
    if not isinstance(args, dict):
        return False, "arguments are not an object"
    for key in schema.get("required", []):
        if key not in args:
            return False, f"missing required {key!r}"
    if not _type_ok(args, schema):
        return False, f"arguments violate schema: {args}"
    for key, needle in (want.get("args") or {}).items():
        if key not in args:
            return False, f"missing expected {key!r}"
        if not _contains(args[key], needle):
            return False, f"{key}={args[key]!r} does not contain {needle!r}"
    return True, ""


def _contains(value, needle) -> bool:
    """Loose match of an expected argument: a list needle needs each element present
    somewhere in the value list, a dict needle needs each key matched, and a scalar
    needle is a case-insensitive substring of the value's text."""
    if isinstance(needle, list):
        return isinstance(value, list) and all(any(_contains(v, n) for v in value) for n in needle)
    if isinstance(needle, dict):
        return isinstance(value, dict) and all(k in value and _contains(value[k], n) for k, n in needle.items())
    return str(needle).lower() in str(value).lower()


SCENARIOS = [
    ("weather_required", "What is the weather in Paris right now? Use the tool.", [{"name": "get_weather", "args": {"city": "Paris"}}]),
    ("weather_enum", "What is the weather in Tokyo in celsius? Use the tool.", [{"name": "get_weather", "args": {"city": "Tokyo", "unit": "celsius"}}]),
    ("search_limit", "Search the web for 'GB10 unified memory benchmarks', limit 5. Use the tool.", [{"name": "search_web", "args": {"query": "GB10", "limit": 5}}]),
    ("email_required", "Email ops@example.com with subject 'Incident' and body 'Resolved'. Use the tool.", [{"name": "send_email", "args": {"to": "ops@example.com", "subject": "Incident"}}]),
    ("email_cc_array", "Email a@example.com, subject 'Hi', body 'Hello', cc b@example.com and c@example.com. Use the tool.", [{"name": "send_email", "args": {"to": "a@example.com", "cc": ["b@example.com"]}}]),
    ("order_integer", "Create an order for customer 42 with 2 units of SKU abc-1. Use the tool.", [{"name": "create_order", "args": {"customer_id": 42}}]),
    ("order_nested", "Create an expedited order for customer 7: one 'pen' and three 'notebook'. Use the tool.", [{"name": "create_order", "args": {"customer_id": 7, "expedited": True}}]),
    ("translate_enum", "Translate 'good morning' to French. Use the tool.", [{"name": "translate", "args": {"text": "good morning", "target_lang": "fr"}}]),
    ("sql_true", "Run a read-only SQL query selecting all users. Use the tool.", [{"name": "run_sql", "args": {"query": "select", "readonly": True}}]),
    ("multi_tool", "Get the weather in Paris and search the web for 'Paris forecast'. Use both tools.", [{"name": "get_weather", "args": {"city": "Paris"}}, {"name": "search_web", "args": {"query": "Paris"}}]),
    ("two_cities", "Get the weather in Tokyo and in Oslo. Use the tool for each city.", [{"name": "get_weather", "args": {"city": "Tokyo"}}, {"name": "get_weather", "args": {"city": "Oslo"}}]),
    ("weather_fahrenheit", "What is the weather in New York in fahrenheit? Use the tool.", [{"name": "get_weather", "args": {"city": "New York", "unit": "fahrenheit"}}]),
    ("search_integer", "Search the web for 'CUDA 13 release notes' with limit 10. Use the tool.", [{"name": "search_web", "args": {"query": "CUDA 13", "limit": 10}}]),
    ("email_body", "Send an email to dev@example.com, subject 'Deploy', body 'The deploy finished'. Use the tool.", [{"name": "send_email", "args": {"to": "dev@example.com", "body": "deploy finished"}}]),
    ("email_cc_two", "Email lead@example.com, subject 'Status', body 'Green', cc pm@example.com and qa@example.com. Use the tool.", [{"name": "send_email", "args": {"to": "lead@example.com", "cc": ["pm@example.com", "qa@example.com"]}}]),
    ("order_two_skus", "Create an order for customer 9 with one 'lamp' and two 'cable'. Use the tool.", [{"name": "create_order", "args": {"customer_id": 9, "items": [{"sku": "lamp"}, {"sku": "cable"}]}}]),
    ("order_standard", "Create a standard order for customer 3 with five 'bolt'. Use the tool.", [{"name": "create_order", "args": {"customer_id": 3, "expedited": False}}]),
    ("translate_german", "Translate 'see you later' to German. Use the tool.", [{"name": "translate", "args": {"text": "see you later", "target_lang": "de"}}]),
    ("translate_spanish", "Translate 'good night' to Spanish. Use the tool.", [{"name": "translate", "args": {"text": "good night", "target_lang": "es"}}]),
    ("sql_false", "Run the SQL query SELECT 1 with readonly set to false. Use the tool.", [{"name": "run_sql", "args": {"query": "SELECT 1", "readonly": False}}]),
    ("three_tools", "Get the weather in Paris, search the web for 'Paris news', and translate 'hello' to French. Use all three tools.", [{"name": "get_weather", "args": {"city": "Paris"}}, {"name": "search_web", "args": {"query": "Paris"}}, {"name": "translate", "args": {"target_lang": "fr"}}]),
    ("search_phrase", "Search the web for 'switchless RoCE ring NCCL tuning'. Use the tool.", [{"name": "search_web", "args": {"query": "switchless RoCE"}}]),
    ("email_no_cc", "Email solo@example.com, subject 'Note', body 'One line'. Do not cc anyone. Use the tool.", [{"name": "send_email", "args": {"to": "solo@example.com", "subject": "Note"}}]),
    ("search_limit_one", "Search the web for 'needle in a haystack eval', limit 1. Use the tool.", [{"name": "search_web", "args": {"query": "needle", "limit": 1}}]),
]


def call(url: str, body: dict, api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--min-pass", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    rows = []

    def record(name: str, ok: bool, detail: str, response: dict | None = None):
        rows.append({"scenario": name, "ok": ok, "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"))
        if not ok and response is not None:
            rows[-1]["response"] = {
                "content": _content(response)[:300],
                "tool_calls": _calls(response),
            }

    for name, prompt, wants in SCENARIOS:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 1024,
            "temperature": 0,
        }
        try:
            response = call(url, body, args.api_key)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            record(name, False, f"request failed: {exc}")
            continue
        calls = _calls(response)
        if not calls:
            record(name, False, f"no tool_calls; content={_content(response)[:100]!r}", response)
            continue
        remaining = list(calls)
        ok, detail = True, ""
        for want in wants:
            match = next((c for c in remaining if (c.get("function") or {}).get("name") == want["name"]), None)
            if match is None:
                ok, detail = False, f"no call to {want['name']}; got {[ (c.get('function') or {}).get('name') for c in calls ]}"
                break
            remaining.remove(match)
            good, why = _judge_call(match, want)
            if not good:
                ok, detail = False, f"{want['name']}: {why}"
                break
        record(name, ok, detail, response)

    # No-tool question: the parser must not invent a call.
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Explain in one sentence why the sky is blue."}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0,
    }
    try:
        response = call(url, body, args.api_key)
        ok = not _calls(response) and len(_content(response)) > 10
        record("no_tool", ok, "unexpected tool call" if _calls(response) else "empty answer", response)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        record("no_tool", False, str(exc))

    # Tool-result follow-up: the model must answer from the tool result.
    weather_call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
    body = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."},
            {"role": "assistant", "content": "", "tool_calls": [weather_call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21, \"conditions\": \"clear\"}"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0,
    }
    try:
        response = call(url, body, args.api_key)
        text = _content(response).lower()
        ok = ("21" in text) and not _calls(response)
        record("tool_result_followup", ok, f"answer did not use the result: {text[:120]!r}", response)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        record("tool_result_followup", False, str(exc))

    # Error recovery: a failed tool result should produce a corrected retry or
    # an explicit explanation, never a silent success claim.
    error_call = {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Atlantis\"}"}}
    body = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "What is the weather in Atlantis? Use the tool."},
            {"role": "assistant", "content": "", "tool_calls": [error_call]},
            {"role": "tool", "tool_call_id": "call_2", "content": "{\"error\": \"unknown city: Atlantis\"}"},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0,
    }
    try:
        response = call(url, body, args.api_key)
        text = _content(response).lower()
        retry = bool(_calls(response))
        honest = bool(re.search(r"unknown|not (found|available)|cannot|can't|no weather", text))
        record("tool_error_recovery", retry or honest,
               f"neither a corrected call nor an honest explanation; content={text[:120]!r}", response)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        record("tool_error_recovery", False, str(exc))

    passed = sum(1 for r in rows if r["ok"])
    print(f"toolgate {passed}/{len(rows)}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"passed": passed, "total": len(rows), "min_pass": args.min_pass, "rows": rows}, f, indent=1)
    return 0 if passed >= args.min_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

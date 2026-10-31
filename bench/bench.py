#!/usr/bin/env python3
"""Speed, long-context and regression client for the MiniMax-M3.1 TP2 server.

Stdlib only; runs from the control machine or the head node.

  python3 bench/bench.py --base-url http://192.0.2.11:8000 --model minimax-m3.1-flash <step>

Steps
  decode   Concurrency sweep x prompt type, temperature 0, fixed --seconds windows.
           Every timed request is one of c streaming workers; tokens come from
           usage.completion_tokens (the engine's own count), never from SSE chunk
           counts: DSpark emits up to k+1 tokens in one delta.
  prefill  One cold request per length (unique nonce), max_tokens 1. Reports the
           actual usage.prompt_tokens, wall time and prompt_tokens/wall.
  needle   A 6-character code at 10% and 50% depth of unique filler. PASS when
           the code appears in content or reasoning.
  spec     /metrics deltas around c=1 decode: accepted tokens per draft step.
  pin      Write the reference: teacher-forced mean NLL of six fixed passages
           (/v1/completions with prompt_logprobs; no generation, so no divergence).
  check    Re-score the passages; exit 1 when a passage's mean NLL moves more than
           --nll-tol (relative, default 0.10) from the reference.
  all      decode, prefill, needle, spec, check.

Thinking: MiniMax's template is adaptive; this client sends no thinking flag
and records usage.completion_tokens_details.reasoning_tokens when exported.
--effort passes reasoning_effort for a fixed effort level.

Results go to bench/runs/<YYYY-mm-dd-HHMMSS>-<step>/ (results.json + summary.md).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DECODE_PROMPTS = {
    "prose": "Write a long, detailed essay on how the invention of the printing press changed European science, religion and politics over the following three centuries.",
    "code": "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, a decorator interface, statistics counters and an extensive unittest suite.",
    "structured": "Produce a detailed Markdown reference guide to TCP congestion control with headings, numbered steps, bullet lists and a comparison table of Reno, CUBIC, BBR and Vegas.",
    "json": "Output a JSON array of 40 fictional library books, each an object with title, author, year, isbn, genres (array), page_count, rating and a two-sentence summary. Output only JSON.",
    "math": "A train leaves city A at 9:00 traveling 80 km/h toward city B, 360 km away. A second train leaves B at 10:00 traveling 100 km/h toward A. Solve step by step, with all arithmetic, and state when and where they meet.",
}

# Teacher-forced passages: the server scores each one token by token. On this
# stack greedy generation is not reproducible even within one boot (near-ties in
# the sparse-attention block selection flip early tokens), so an exact token-id
# pin cannot gate anything. Per-passage mean NLL moved <= 2.4% across repeats and
# across two Spark pairs (2026-10-24); wrong or corrupt weights move it by far more.
NLL_PASSAGES = [
    ("prose", "The Industrial Revolution began in Britain in the late eighteenth century. Steam engines, first used "
     "to pump water out of mines, were soon adapted to drive textile machinery, and later locomotives and ships. "
     "Cities grew quickly as workers moved from farms to factories, and new problems of sanitation, housing and "
     "labour conditions followed. Reformers documented the conditions in mills and mines, and parliament passed "
     "the first factory acts, limiting the hours that children could work."),
    ("code", "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
     "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
     "            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n\n"
     "def is_sorted(xs):\n    return all(x <= y for x, y in zip(xs, xs[1:]))\n\n"
     "assert is_sorted(merge_sorted([1, 4, 9], [2, 3, 10]))\n"),
    ("math", "Question: A train travels 180 kilometres in 2 hours and then 120 kilometres in 1.5 hours. What is its "
     "average speed for the whole journey?\nAnswer: The total distance is 300 kilometres and the total time is 3.5 "
     "hours, so the average speed is 300 / 3.5 = 85.7 kilometres per hour."),
    ("json", '{"name": "Ada Lovelace", "born": 1815, "city": "London", "known_for": ["Analytical Engine notes", '
     '"first published algorithm"], "collaborator": "Charles Babbage"}'),
    ("reasoning", "All squares are rectangles, and all rectangles have four right angles. A rhombus has four equal "
     "sides but not necessarily right angles. Therefore every square is a rhombus with right angles, but not every "
     "rhombus is a square."),
    ("dialogue", "User: What is the capital of France?\nAssistant: The capital of France is Paris.\nUser: And of "
     "Japan?\nAssistant: The capital of Japan is Tokyo."),
]

WORDS = (
    "almanac beacon copper delta ember fathom gable harbour isotope juniper kelp lantern meadow "
    "nectar orchard pollen quarry ridge sable tundra umber vessel willow xenon yarrow zephyr"
).split()


def rng_words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def salted_filler(target_words: int, nonce: str, depth: int) -> tuple[str, str]:
    rng = random.Random(f"{nonce}:{target_words}:{depth}")
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    head = rng_words(rng, max(1, target_words * depth // 100))
    tail = rng_words(rng, max(1, target_words - target_words * depth // 100))
    text = f"{nonce} {head}\n\nThe access code is {code}.\n\n{tail}"
    return text, code


def api_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        return base + path[3:]
    return base + path


def post_json(url: str, body: dict, api_key: str, timeout: int = 900) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_text(url: str, api_key: str, timeout: int = 30) -> str:
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def stream_chat(url: str, body: dict, api_key: str, timeout: int = 900) -> dict:
    """One streaming chat request. Returns wall, ttft, token timestamps, usage, text."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    body = dict(body)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    started = time.time()
    ttft = None
    last_delta = None
    chunks = 0
    text_parts = []
    reasoning_parts = []
    usage = None
    finish_reason = None
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
                    text_parts.append(piece)
                if reason:
                    reasoning_parts.append(reason)
                if piece or reason:
                    now = time.time()
                    if ttft is None:
                        ttft = now - started
                    last_delta = now
                    chunks += 1
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    ended = time.time()
    return {
        "wall_s": ended - started,
        "ttft_s": ttft,
        "last_delta_s": (last_delta - started) if last_delta else None,
        "deltas": chunks,
        "usage": usage,
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
    }


def max_model_len(base: str, api_key: str, model: str) -> int | None:
    try:
        payload = post_json(api_url(base, "/v1/models"), {}, api_key)
    except Exception:
        try:
            payload = json.loads(get_text(api_url(base, "/v1/models"), api_key))
        except Exception:
            return None
    for row in payload.get("data") or []:
        if row.get("id") == model or len(payload.get("data") or []) == 1:
            value = row.get("max_model_len")
            return int(value) if value else None
    return None


def metrics(base: str, api_key: str) -> dict:
    try:
        text = get_text(api_url(base, "/metrics"), api_key)
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{")[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def spec_metrics(before: dict, after: dict) -> dict:
    def delta(name):
        return after.get(name, 0.0) - before.get(name, 0.0)

    drafts = delta("vllm:spec_decode_num_drafts_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total")
    out = {
        "drafts": drafts,
        "accepted_tokens": accepted,
        "draft_tokens": draft_tokens,
        "tokens_per_step": (accepted / drafts) if drafts else None,
        "accept_rate": (accepted / draft_tokens) if draft_tokens else None,
    }
    if not drafts:
        out["available"] = sorted(k for k in after if "spec_decode" in k)
    return out


def decode_cell(args, prompt_type: str, concurrency: int, out_dir: str) -> dict:
    url = api_url(args.base_url, "/v1/chat/completions")
    prompt = DECODE_PROMPTS[prompt_type]
    deadline = None
    rows = []
    errors = []
    lock = threading.Lock()

    def worker(_):
        local = []
        local_errors = []
        while time.time() < deadline:
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }
            if args.effort:
                body["reasoning_effort"] = args.effort
            try:
                row = stream_chat(url, body, args.api_key, timeout=args.timeout)
                usage = row["usage"] or {}
                completion = usage.get("completion_tokens")
                if completion is None:
                    completion = max(0, row["deltas"])
                decode_s = None
                if row["last_delta_s"] is not None and row["ttft_s"] is not None:
                    decode_s = max(1e-6, row["last_delta_s"] - row["ttft_s"])
                local.append(
                    {
                        "completion_tokens": completion,
                        "wall_s": row["wall_s"],
                        "end_s": time.time() - cell_start,
                        "ttft_s": row["ttft_s"],
                        "decode_s": decode_s,
                        "tok_s": ((completion - 1) / decode_s) if decode_s and completion > 1 else None,
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "finish_reason": row["finish_reason"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one failed request must not kill the cell
                local_errors.append(str(exc))
        with lock:
            rows.extend(local)
            errors.extend(local_errors)

    cell_start = time.time()
    deadline = cell_start + args.seconds
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(worker, range(concurrency)))
    # Aggregate = every token the cell produced over the cell's elapsed time (until its
    # last request finished), not over one request's duration.
    elapsed = max((r["end_s"] for r in rows), default=0.0)
    total_tokens = sum(r["completion_tokens"] for r in rows)
    tok_s = [r["tok_s"] for r in rows if r["tok_s"]]
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    cell = {
        "concurrency": concurrency,
        "prompt_type": prompt_type,
        "requests": len(rows),
        "errors": len(errors),
        "error_sample": errors[:3],
        "agg_tok_s": total_tokens / max(1e-6, elapsed),
        "per_stream_tok_s_median": round(statistics.median(tok_s), 2) if tok_s else None,
        "ttft_s_median": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_s_p90": round(sorted(ttfts)[int(0.9 * (len(ttfts) - 1))], 3) if ttfts else None,
        "reasoning_tokens_median": (
            round(statistics.median([r["reasoning_tokens"] for r in rows if r["reasoning_tokens"]]), 1)
            if any(r["reasoning_tokens"] for r in rows) else None
        ),
    }
    print(
        f"  c={concurrency:<2} {prompt_type:<10} {cell['agg_tok_s']:7.1f} tok/s agg"
        f"  per-stream {cell['per_stream_tok_s_median'] or 0:6.2f}"
        f"  TTFT {cell['ttft_s_median'] or 0:6.2f}s  req {cell['requests']:3d}  err {cell['errors']}",
        flush=True,
    )
    return cell


def step_decode(args, out_dir: str) -> dict:
    concs = [int(x) for x in args.conc.split(",") if x.strip()]
    types = [x for x in args.types.split(",") if x.strip()]
    cells = []
    print(f"decode: {args.seconds}s windows, temperature 0, max_tokens {args.max_tokens}", flush=True)
    for c in concs:
        for prompt_type in types:
            cells.append(decode_cell(args, prompt_type, c, out_dir))
    return {"cells": cells, "seconds": args.seconds, "max_tokens": args.max_tokens}


def step_prefill(args, out_dir: str) -> dict:
    limit = max_model_len(args.base_url, args.api_key, args.model)
    lengths = [parse_size(x) for x in args.lengths.split(",") if x.strip()]
    rows = []
    print(f"prefill: lengths {lengths} (max_model_len {limit})", flush=True)
    for target in lengths:
        if limit and target > limit:
            print(f"  {target}: skipped (> max_model_len)")
            continue
        nonce = uuid.uuid4().hex[:10]
        words = max(16, int(target / args.word_ratio))
        text = f"Nonce {nonce}. " + rng_words(random.Random(nonce), words) + "\n\nSay OK."
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0,
        }
        try:
            row = stream_chat(api_url(args.base_url, "/v1/chat/completions"), body, args.api_key, timeout=args.timeout)
            usage = row.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            if not prompt_tokens:
                prompt_tokens = post_json(
                    api_url(args.base_url, "/tokenize"),
                    {"model": args.model, "messages": body["messages"], "add_generation_prompt": True},
                    args.api_key,
                ).get("count")
            wall = row["wall_s"]
            rows.append(
                {
                    "target_words": target,
                    "prompt_tokens": prompt_tokens,
                    "wall_s": round(wall, 3),
                    "tok_s": round(prompt_tokens / wall, 1) if prompt_tokens and wall else None,
                    "ttft_s": round(row["ttft_s"], 3) if row["ttft_s"] else None,
                }
            )
            print(f"  {target:>8} target -> {prompt_tokens} tokens  {rows[-1]['tok_s']} tok/s  TTFT {rows[-1]['ttft_s']}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"target_words": target, "error": str(exc)})
            print(f"  {target}: FAILED {exc}", flush=True)
    return {"rows": rows}


def step_needle(args, out_dir: str) -> dict:
    limit = max_model_len(args.base_url, args.api_key, args.model) or args.tokens
    target = min(args.tokens, max(1024, limit - 1024))
    rows = []
    print(f"needle: {target} tokens, depths {args.depths}", flush=True)
    for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
        nonce = uuid.uuid4().hex[:10]
        words = max(64, int(target / args.word_ratio))
        text, code = salted_filler(words, nonce, depth)
        body = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": text + f"\n\nWhat is the access code? Reply with the code only."}
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        try:
            row = post_json(api_url(args.base_url, "/v1/chat/completions"), body, args.api_key, timeout=args.timeout)
            usage = row.get("usage") or {}
            message = (row.get("choices") or [{}])[0].get("message") or {}
            haystack = (message.get("content") or "") + (message.get("reasoning") or message.get("reasoning_content") or "")
            ok = code in haystack
            rows.append(
                {
                    "depth_pct": depth,
                    "code": code,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "ok": ok,
                    "answer": (message.get("content") or "")[:160],
                }
            )
            print(f"  depth {depth:>2}%  prompt {usage.get('prompt_tokens')}  {'PASS' if ok else 'FAIL'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"depth_pct": depth, "ok": False, "error": str(exc)})
    return {"rows": rows, "target_tokens": target}


def step_spec(args, out_dir: str) -> dict:
    rows = []
    types = [x for x in (args.spec_types or "prose,code").split(",") if x.strip()]
    for prompt_type in types:
        before = metrics(args.base_url, args.api_key)
        cell = decode_cell(args, prompt_type, 1, out_dir)
        after = metrics(args.base_url, args.api_key)
        row = {"prompt_type": prompt_type, "cell": cell, "spec": spec_metrics(before, after)}
        rows.append(row)
        spec = row["spec"]
        print(
            f"  {prompt_type:<10} accepted/step {spec['tokens_per_step']}  accept {spec['accept_rate']}",
            flush=True,
        )
    return {"rows": rows}


def _passage_nll(args, text: str) -> dict:
    body = {"model": args.model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    payload = post_json(api_url(args.base_url, "/v1/completions"), body, args.api_key, timeout=args.timeout)
    entries = (payload.get("choices") or [{}])[0].get("prompt_logprobs")
    if not entries:
        raise SystemExit("the server returned no prompt_logprobs; this vLLM build must support them")
    lps = [v["logprob"] for e in entries[1:] if e for v in e.values()]
    return {"tokens": len(lps), "mean_nll": round(-sum(lps) / len(lps), 6)}


def step_pin(args, out_dir: str) -> dict:
    rows = []
    for label, text in NLL_PASSAGES:
        r = _passage_nll(args, text)
        rows.append({"label": label, **r})
        print(f"  pinned {label}: {r['tokens']} tokens, mean NLL {r['mean_nll']:.4f}", flush=True)
    ref = {"model": args.model, "kind": "teacher-forced-nll", "passages": rows}
    path = args.reference or os.path.join(HERE, "reference.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ref, f, indent=1)
    print(f"wrote {path}")
    return {"reference": path, "passages": len(rows)}


def step_check(args, out_dir: str) -> dict:
    path = args.reference or os.path.join(HERE, "reference.json")
    with open(path) as f:
        ref = json.load(f)
    if ref.get("kind") != "teacher-forced-nll":
        raise SystemExit(f"{path} is not a teacher-forced NLL reference; re-run pin")
    want = {r["label"]: r for r in ref["passages"]}
    rows = []
    failed = 0
    for label, text in NLL_PASSAGES:
        r = _passage_nll(args, text)
        w = want[label]
        rel = abs(r["mean_nll"] - w["mean_nll"]) / w["mean_nll"]
        ok = r["tokens"] == w["tokens"] and rel <= args.nll_tol
        failed += 0 if ok else 1
        rows.append({"label": label, "ok": ok, "mean_nll": r["mean_nll"], "ref_mean_nll": w["mean_nll"],
                     "rel_diff": round(rel, 4)})
        print(f"  {label:<10} {'MATCH' if ok else 'DRIFT'}  mean NLL {r['mean_nll']:.4f} vs {w['mean_nll']:.4f} "
              f"({100 * rel:.2f}%)", flush=True)
    return {"rows": rows, "failed": failed, "tolerance": args.nll_tol}


def write_summary(out_dir: str, step: str, data: dict, args) -> None:
    lines = [f"# {step} — {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if step == "decode":
        lines += ["| c | type | agg tok/s | per-stream tok/s | TTFT med | requests | errors |", "|---|---|---|---|---|---|---|"]
        for c in data["cells"]:
            lines.append(
                f"| {c['concurrency']} | {c['prompt_type']} | {c['agg_tok_s']:.1f} | "
                f"{c['per_stream_tok_s_median'] or 0:.2f} | {c['ttft_s_median'] or 0:.2f} | {c['requests']} | {c['errors']} |"
            )
    elif step == "prefill":
        lines += ["| target words | prompt tokens | tok/s | TTFT s |", "|---|---|---|---|"]
        for r in data["rows"]:
            if "error" in r:
                lines.append(f"| {r['target_words']} | error | | |")
            else:
                lines.append(f"| {r['target_words']} | {r['prompt_tokens']} | {r['tok_s']} | {r['ttft_s']} |")
    elif step == "needle":
        lines += ["| depth % | prompt tokens | result |", "|---|---|---|"]
        for r in data["rows"]:
            lines.append(f"| {r['depth_pct']} | {r.get('prompt_tokens')} | {'PASS' if r['ok'] else 'FAIL'} |")
    elif step == "spec":
        lines += ["| prompt | accepted draft tokens/step | accept rate | agg tok/s |", "|---|---|---|---|"]
        for r in data["rows"]:
            s = r["spec"]
            lines.append(
                f"| {r['prompt_type']} | {s['tokens_per_step']} | {s['accept_rate']} | {r['cell']['agg_tok_s']:.1f} |"
            )
    elif step == "check":
        lines += [f"| passage | result | mean NLL | reference | diff (tol {data['tolerance']:.0%}) |", "|---|---|---|---|---|"]
        for r in data["rows"]:
            lines.append(f"| {r['label']} | {'MATCH' if r['ok'] else 'DRIFT'} | {r['mean_nll']:.4f} | "
                         f"{r['ref_mean_nll']:.4f} | {100 * r['rel_diff']:.2f}% |")
    else:
        lines.append("```json")
        lines.append(json.dumps(data, indent=1)[:20000])
        lines.append("```")
    lines += ["", f"server `{args.base_url}`, model `{args.model}`"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_size(text: str) -> int:
    text = text.strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    if text.endswith("m"):
        return int(float(text[:-1]) * 1024 * 1024)
    return int(text)


def default_out(step: str) -> str:
    return os.path.join(HERE, "runs", time.strftime("%Y-%m-%d-%H%M%S") + "-" + step)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["decode", "prefill", "needle", "spec", "pin", "check", "all"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="minimax-m3.1-flash")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", help="run directory")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--types", default="prose,code,structured,json")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--lengths", default="8k,32k,128k")
    ap.add_argument("--tokens", type=int, default=131072, help="needle target prompt tokens")
    ap.add_argument("--depths", default="10,50")
    ap.add_argument("--spec-types", default="prose,code")
    ap.add_argument("--effort", default=None, help="reasoning_effort for every bench request")
    ap.add_argument("--word-ratio", type=float, default=1.3, help="tokens per generated word (MiniMax ~1.3)")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--nll-tol", type=float, default=0.10, help="check: max relative change of a passage's mean NLL")
    args = ap.parse_args()

    steps = ["decode", "prefill", "needle", "spec", "check"] if args.step == "all" else [args.step]
    for step in steps:
        out_dir = args.out if (args.out and len(steps) == 1) else default_out(step)
        os.makedirs(out_dir, exist_ok=True)
        started = time.time()
        if step == "decode":
            data = step_decode(args, out_dir)
        elif step == "prefill":
            data = step_prefill(args, out_dir)
        elif step == "needle":
            data = step_needle(args, out_dir)
        elif step == "spec":
            data = step_spec(args, out_dir)
        elif step == "pin":
            data = step_pin(args, out_dir)
        else:
            data = step_check(args, out_dir)
        result = {
            "step": step,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - started, 1),
            "server": args.base_url,
            "model": args.model,
            "result": data,
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=1)
        write_summary(out_dir, step, data, args)
        print(f"{step}: {out_dir}")
        if step == "check" and data.get("failed"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

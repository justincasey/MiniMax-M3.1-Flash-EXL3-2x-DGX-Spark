#!/usr/bin/env python3
"""Boot-phase breakdown from one `./m31.sh start` run directory.

Reads <run>/start.txt (controller timings, t_launch) and <run>/rank{0,1}.log
(`docker logs -t`, RFC 3339 timestamps) and writes <run>/boot_phases.json:

  - controller phases: preflight_s, engine_up_s, warmup_s, smoke_s, ready_s;
  - per rank: every engine milestone found (seconds since launch, plus the
    duration the engine itself reports), and the largest silent gaps between
    consecutive log lines (the unlabelled stalls).

  python3 scripts/boot_phases.py --run runs/<ts> [--metrics]

--metrics prints `METRIC boot_<name>=<seconds>` lines for the harness.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

# name -> (pattern, index of the reported-duration group or None)
MILESTONES = {
    "first_log": (r".", None),
    "dist_init": (r"world_size=2 rank=\d|Initializing .*distributed|init_distributed_environment", None),
    "weights_load": (r"Loading weights took ([\d.]+) ?s", 1),
    "model_load": (r"Model loading took [\d.]+ ?GiB(?: memory)? and ([\d.]+) ?s", 1),
    "dynamo": (r"Dynamo bytecode transform time: ([\d.]+) ?s", 1),
    "compile": (r"torch\.compile takes ([\d.]+) ?s in total", 1),
    "profile": (r"Memory profiling takes ([\d.]+) ?s", 1),
    "kv_alloc": (r"GPU KV cache size|KV cache memory|kv cache size", None),
    "graph_capture": (r"Graph capturing finished in ([\d.]+) ?s", 1),
    "init_engine": (r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) ?s", 1),
    "api_start": (r"Starting vLLM API server|Starting vLLM server", None),
    "app_ready": (r"Application startup complete", None),
}
TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d) (.*)$")


def parse_ts(stamp: str, frac: str | None, zone: str) -> float:
    digits = (frac or ".")[1:7].ljust(6, "0")
    zone = "+00:00" if zone == "Z" else zone
    return dt.datetime.fromisoformat(f"{stamp}.{digits}{zone}").timestamp()


def read_start(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


def rank_phases(path: str, t_launch: float) -> dict:
    lines = []
    with open(path, errors="replace") as f:
        for raw in f:
            m = TS.match(raw.rstrip("\n"))
            if m:
                lines.append((parse_ts(m.group(1), m.group(2), m.group(3)), m.group(4)))
    found = {}
    for name, (pattern, group) in MILESTONES.items():
        rx = re.compile(pattern)
        for t, text in lines:
            m = rx.search(text)
            if m:
                row = {"at_s": round(t - t_launch, 1)}
                if group:
                    row["took_s"] = float(m.group(group))
                found[name] = row
                if name != "kv_alloc":
                    break
    gaps = []
    for (t0, _), (t1, text) in zip(lines, lines[1:]):
        if t1 - t0 >= 2.0:
            gaps.append({"at_s": round(t0 - t_launch, 1), "gap_s": round(t1 - t0, 1), "next": text[:160]})
    gaps.sort(key=lambda g: -g["gap_s"])
    last = round(lines[-1][0] - t_launch, 1) if lines else None
    return {"milestones": found, "largest_gaps": gaps[:10], "lines": len(lines), "last_log_s": last}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--metrics", action="store_true")
    args = ap.parse_args()
    start = read_start(os.path.join(args.run, "start.txt"))
    t_launch = float(start["t_launch"])
    out = {"controller": {k: float(start[k]) for k in
                          ("preflight_s", "endpoint_up_s", "engine_up_s", "warmup_s", "smoke_s", "ready_s")
                          if k in start}}
    for r in (0, 1):
        log = os.path.join(args.run, f"rank{r}.log")
        if os.path.exists(log):
            out[f"rank{r}"] = rank_phases(log, t_launch)
    with open(os.path.join(args.run, "boot_phases.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    if args.metrics:
        for k, v in out["controller"].items():
            print(f"METRIC {k}={v}")
        for name, row in out.get("rank0", {}).get("milestones", {}).items():
            print(f"METRIC boot_{name}_at_s={row['at_s']}")
            if "took_s" in row:
                print(f"METRIC boot_{name}_took_s={row['took_s']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

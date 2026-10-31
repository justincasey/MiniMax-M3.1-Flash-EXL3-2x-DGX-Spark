#!/usr/bin/env python3
"""Census a MiniMax-M3.1 EXL3 pack: per-module dtypes, shapes and bytes.

Reads only safetensors headers (a local pack, or HTTP range requests against a
pinned Hub revision), so it never downloads weights.

  # Maintainers: regenerate the committed census from the pinned revision
  python3 scripts/pack_census.py --repo vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw \
      --revision 99bf3a57af5d07245ea9abb5426e7dce6af6356b -o config/lanes/2.15bpw/pack-census.json

  # Runtime: census a fetched pack (local headers, no network)
  python3 scripts/pack_census.py --pack ~/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw

The census is the byte/structure authority for tests/test_pack_layout.py. It is deterministic: the same pack
revision always produces the same JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from collections import defaultdict

SUFFIXES = (".weight_scale_inv", ".weight", ".trellis", ".suh", ".svh", ".mul1", ".bias")
ROUTED = ".block_sparse_moe.experts."


def _read_header_local(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def _read_header_http(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        header = json.loads(r.read())
    header.pop("__metadata__", None)
    return header


def _module_of(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _suffix_of(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return suffix.lstrip(".")
    return "tensor"


def census(entries: list[tuple[str, str, dict]], repo: str, revision: str) -> dict:
    """entries: [(file, tensor_name, {"dtype","shape","data_offsets"})]"""
    modules: dict[str, dict] = {}
    routed: dict[str, int] = defaultdict(int)
    dtypes: dict[str, int] = defaultdict(int)
    files: dict[str, int] = defaultdict(int)

    for filename, name, meta in entries:
        size = meta["data_offsets"][1] - meta["data_offsets"][0]
        dtypes[meta["dtype"]] += size
        files[filename] += size
        suffix = _suffix_of(name)
        if ROUTED in name:
            # Collapse every expert of a layer's expert block, keeping the
            # projection (w1/w2/w3) and tensor suffix distinct.
            root, rest = name.split(ROUTED, 1)
            projection = rest.split(".", 2)[1] if "." in rest else "*"
            key = f"{root}{ROUTED}*.{projection}.{suffix}"
            routed[key] += size
            continue
        mod = _module_of(name)
        slot = modules.setdefault(mod, {"suffixes": {}, "bytes": 0})
        slot["suffixes"][suffix] = {
            "dtype": meta["dtype"],
            "shape": list(meta["shape"]),
            "bytes": size,
        }
        slot["bytes"] += size

    # Routed entries are aggregate rows, not modules; keep them separate.
    out_modules = {}
    for mod in sorted(modules):
        out_modules[mod] = {
            "bytes": modules[mod]["bytes"],
            "tensors": {k: modules[mod]["suffixes"][k] for k in sorted(modules[mod]["suffixes"])},
        }
    out_routed = {k: routed[k] for k in sorted(routed)}
    return {
        "repo": repo,
        "revision": revision,
        "total_bytes": sum(files.values()),
        "files": {k: files[k] for k in sorted(files)},
        "dtypes": {k: dtypes[k] for k in sorted(dtypes)},
        "modules": out_modules,
        "routed_tensors": out_routed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", help="local model directory (config + index + safetensors)")
    src.add_argument("--repo", help="Hub repo id for HTTP header reads")
    ap.add_argument("--revision", help="Hub revision (required with --repo)")
    ap.add_argument("-o", "--out", help="write JSON here (default: stdout)")
    args = ap.parse_args()

    if args.pack:
        index_path = os.path.join(args.pack, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        with open(os.path.join(args.pack, "config.json")) as f:
            config = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        entries = []
        for shard in shards:
            header = _read_header_local(os.path.join(args.pack, shard))
            for name, meta in header.items():
                entries.append((shard, name, meta))
        repo = os.path.basename(os.path.abspath(args.pack))
        revision = str(config.get("_name_or_path", "local"))
    else:
        if not args.revision:
            ap.error("--repo requires --revision")
        base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
        index_url = base + "model.safetensors.index.json"
        with urllib.request.urlopen(index_url, timeout=60) as r:
            index = json.load(r)
        shards = sorted(set(index["weight_map"].values()))
        entries = []
        for i, shard in enumerate(shards, 1):
            header = _read_header_http(base + shard)
            for name, meta in header.items():
                entries.append((shard, name, meta))
            print(f"[{i}/{len(shards)}] {shard}", file=sys.stderr)
        repo = args.repo
        revision = args.revision

    out = census(entries, repo, revision)
    payload = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        print(f"wrote {args.out} ({len(payload)} bytes, sha256 {digest})", file=sys.stderr)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify a fetched pack against the pinned manifest.

Manifest rows are ``<sha256>\\t<bytes>\\t<path>`` (config/lanes/2.15bpw/pack-manifest.tsv).
Byte authority is the sha256; the size check is the fast pre-pass.

  python3 scripts/verify_manifest.py --pack ~/models/<pack> --manifest config/lanes/2.15bpw/pack-manifest.tsv
  ... --quick          # sizes only (doctor)
  ... --threads 8 --json out.json
  ... --allow-extra    # ignore files not in the manifest (HF metadata cache)

Exit 0 only when every file exists, has the pinned size and (unless --quick)
the pinned sha256. Missing/extra/mismatched files are listed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time


def sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--quick", action="store_true", help="size check only")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--allow-extra", action="store_true", default=True)
    args = ap.parse_args()

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, size, path = line.split("\t")
            rows.append((digest, int(size), path))
    if not rows:
        raise SystemExit("empty manifest")

    expected = {path for _, _, path in rows}
    missing = []
    wrong_size = []
    present = []
    for digest, size, path in rows:
        full = os.path.join(args.pack, path)
        if not os.path.isfile(full):
            missing.append(path)
            continue
        actual = os.path.getsize(full)
        if actual != size:
            wrong_size.append({"path": path, "want": size, "got": actual})
            continue
        present.append((digest, path))

    bad_hash = []
    started = time.time()
    if not args.quick and present:
        done = 0

        def check(item):
            digest, path = item
            return path, sha256(os.path.join(args.pack, path)), digest

        with cf.ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
            futures = [ex.submit(check, item) for item in present]
            for fut in cf.as_completed(futures):
                path, got, want = fut.result()
                done += 1
                if got != want:
                    bad_hash.append({"path": path, "want": want, "got": got})
                if done % 8 == 0 or done == len(present):
                    print(f"  hashed {done}/{len(present)} ({time.time() - started:.0f}s)", file=sys.stderr)

    extra = []
    if not args.allow_extra:
        for root, dirs, files in os.walk(args.pack):
            dirs[:] = [d for d in dirs if d not in (".cache",)]
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), args.pack)
                if rel not in expected and not rel.endswith(".lock"):
                    extra.append(rel)

    report = {
        "pack": os.path.abspath(args.pack),
        "manifest": os.path.abspath(args.manifest),
        "files": len(rows),
        "bytes": sum(size for _, size, _ in rows),
        "quick": args.quick,
        "seconds": round(time.time() - started, 1),
        "missing": missing,
        "wrong_size": wrong_size,
        "bad_hash": bad_hash,
        "extra": extra,
        "ok": not (missing or wrong_size or bad_hash),
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    if report["ok"]:
        print(f"OK: {len(rows)} files, {report['bytes'] / 2**30:.1f} GiB verified{'' if not args.quick else ' (sizes only)'}")
        return 0
    if missing:
        print(f"MISSING {len(missing)}:", file=sys.stderr)
        for p in missing[:20]:
            print(f"  {p}", file=sys.stderr)
    if wrong_size:
        print(f"WRONG SIZE {len(wrong_size)}:", file=sys.stderr)
        for row in wrong_size[:20]:
            print(f"  {row['path']}: {row['got']} != {row['want']}", file=sys.stderr)
    if bad_hash:
        print(f"BAD SHA256 {len(bad_hash)}:", file=sys.stderr)
        for row in bad_hash[:20]:
            print(f"  {row['path']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

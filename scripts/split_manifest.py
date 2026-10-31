#!/usr/bin/env python3
"""Split the pack manifest into N byte-balanced parts (file paths only).

  python3 scripts/split_manifest.py --manifest config/lanes/2.15bpw/pack-manifest.tsv --parts 2 --out-dir runs/x/parts

Whole files only, largest first, each to the currently lightest part. The
result is deterministic and each part is resumable: the downloader is given
exact paths with ``hf download --include``.
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--parts", type=int, default=2)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.parts < 1:
        raise SystemExit("--parts must be >= 1")

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sha, size, path = line.split("\t")
            rows.append((int(size), path))
    rows.sort(key=lambda r: (-r[0], r[1]))

    parts = [[] for _ in range(args.parts)]
    totals = [0] * args.parts
    for size, path in rows:
        i = min(range(args.parts), key=lambda j: (totals[j], j))
        parts[i].append(path)
        totals[i] += size

    os.makedirs(args.out_dir, exist_ok=True)
    for i, files in enumerate(parts):
        out = os.path.join(args.out_dir, f"part{i}.txt")
        with open(out, "w") as f:
            f.write("\n".join(files) + ("\n" if files else ""))
        print(f"{out}: {len(files)} files, {totals[i] / 2**30:.1f} GiB")
    print(json.dumps({"parts": [{"files": len(p), "bytes": t} for p, t in zip(parts, totals)]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""patch 0020 correctness: the v2 q4 slice planner must reproduce the reference plan exactly.

  python3 tests/test_q4kv_slice_planner.py          # small cases, any host with torch
  python3 tests/test_q4kv_slice_planner.py --all    # served geometry too (needs a GPU)

Reference = ``vllm_exl3.minimax_q4kv_reference.plan_prefill_slices_fast``, the exact object
the served prefill path calls (``minimax_q4kv.py``). Candidate =
``vllm_exl3.slice_planner.plan_prefill_slices_v2``. Both are loaded from the build context
that ``scripts/prepare_src.sh`` stages with both patch series applied, so this pins the
shipped files rather than a copy: the build context is re-staged into a fresh temp dir on
every run and its digest is checked against ``VALIDATED_TREES`` before anything is imported.
Needs network for the pinned commits, cached under ``.work/test-src`` by
``tests/test_pack_layout.py``. A stale ``VLLM_EXL3_SRC`` override skips that staging and the
digest check, so only set it deliberately.

Every case asserts the same slice count, the same ``reqs``, the same ``q`` ranges, the same
page count per slice and element-wise identical page id lists in the same order. A plan
difference here is a correctness failure, not a performance result: the staged bf16 bytes,
and therefore the attention output, are functions of this plan alone.

Cases cover the extremes: empty selections (every entry ``-1``), single rows, single pages,
duplicate pages inside a row and across rows, structurally identical rows (the tie case for
the greedy), caps at 1 / at the union size / one below it (the hard floor), caps at or above
``nb`` (the branch where no split is needed), partial blocks at the causal edge (``nb``
smaller than ``topk``), many requests, non-block-multiple prefixes, and the served pool
width (``nb = 9,042`` pages per rank per layer at ``KVB=20 GB`` over 60 layers).

The served-geometry cases allocate the real 333 MB pool and run the reference planner's
``[rows, nb]`` int64 passes (about 1.2 GiB per call); on CPU that is minutes per case, so
they need ``--all``, a GPU, and - if anything is serving on that GPU - a capped container.
Phase 3 ran them on the GB10 pair (kernel-lab/phase3/records, branch kernel-lab-phase3).

``greedy_row_slices``' ``max_pages >= nb`` branch is not reachable from
``plan_prefill_slices_v2`` (it is called only when one request's union already exceeds the
cap, and a union is at most ``nb`` pages), so the ``cap >= nb`` case below pins the
reference's no-split path, not that branch; the branch stays only so the helper is safe on
its own.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import torch  # noqa: E402

FAILS = []
DEV = "cpu"
_STAGE = []
KVH = 2
TOPK = 16
BLOCK = 128


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILS.append(name)
        print(f"FAIL {name}" + (f": {detail}" if detail else ""))


def staged_digest(root: str) -> str:
    """``test_pack_layout.tree_digest`` over staged sources only.

    Byte-identical to it for a freshly staged tree, but it skips ``__pycache__`` and
    ``*.pyc``: this test imports the staged modules, and the bytecode that leaves behind
    must not read as a source change. ``VALIDATED_TREES`` is a digest of sources.
    """
    rows = []
    for d, _, files in os.walk(root):
        if "__pycache__" in d.split(os.sep):
            continue
        for f in files:
            if f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            x = "x" if os.stat(p).st_mode & 0o100 else "-"
            rows.append(f"{h} {x} {os.path.relpath(p, root)}\n".encode())
    return hashlib.sha256(b"".join(sorted(rows))).hexdigest()


def staged_src() -> str:
    """A freshly staged plugin source tree (both patch series applied), digest-checked.

    Staged into a new temp dir on every run - never reused from a cache - and compared
    against ``VALIDATED_TREES`` before anything is imported. A cached tree that merely
    agrees with itself would compare an old candidate against an old reference and print
    PASS forever after an edit to patch 0020: a false green on the only gate for the
    semantics-preserving claim. Fetching is cheap here because the pinned commits are
    cached under ``.work/test-src`` by ``tests/test_pack_layout.py``.

    ``VLLM_EXL3_SRC`` overrides this for running against a tree staged elsewhere; it skips
    the digest check, so use it knowingly.
    """
    import shutil
    import tempfile

    from test_pack_layout import VALIDATED_TREES, pins, run_prepare_src
    env_src = os.environ.get("VLLM_EXL3_SRC")
    if env_src:
        return os.path.join(env_src, "src", "vllm_exl3")
    tmp = tempfile.mkdtemp(prefix="m31-a3-stage-")
    _STAGE.append(tmp)
    out = run_prepare_src(pins(), os.path.join(tmp, "build"))
    if out.returncode != 0:
        raise SystemExit(f"prepare_src failed: {out.stderr[-600:]}")
    tree = os.path.join(tmp, "build", "docker", "vllm-exl3-src")
    got = staged_digest(tree)
    if got != VALIDATED_TREES["vllm-exl3-src"]:
        raise SystemExit(
            f"staged vllm-exl3-src digest {got} != the hardware-validated "
            f"{VALIDATED_TREES['vllm-exl3-src']}: patch 0020 (or a pin) changed without "
            f"re-validating and updating VALIDATED_TREES in tests/test_pack_layout.py")
    return os.path.join(tree, "src", "vllm_exl3")


def load_planners(src: str):
    """Import minimax_q4kv_reference and slice_planner without vllm_exl3/__init__.py.

    Both modules are stdlib + torch only, but ``slice_planner`` does
    ``from vllm_exl3 import minimax_q4kv_reference``, so a synthetic parent package is
    registered first. That also keeps the staged tree's modules ahead of any installed
    copy inside the serving image.
    """
    pkg = types.ModuleType("vllm_exl3")
    pkg.__path__ = [src]
    sys.modules["vllm_exl3"] = pkg
    mods = {}
    for name in ("minimax_q4kv_reference", "slice_planner"):
        spec = importlib.util.spec_from_file_location(f"vllm_exl3.{name}",
                                                      os.path.join(src, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"vllm_exl3.{name}"] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, name, mod)
        mods[name] = mod
    return mods["minimax_q4kv_reference"], mods["slice_planner"]


def make_case(ref, nb, lens, prefixes, seed, mode="normal", topk=TOPK):
    """A q4 pool, a block table and a topk_idx in the shape the served path passes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    total_q = sum(lens)
    pool = torch.randint(0, 256, (nb, 2 * KVH, BLOCK, ref.ROW_BYTES),
                         dtype=torch.uint8, generator=g).to(DEV)
    seqs = [p + l for p, l in zip(prefixes, lens)]
    max_blk = max((s + BLOCK - 1) // BLOCK for s in seqs)
    bt = torch.zeros(len(lens), max_blk, dtype=torch.int32)
    for r, s in enumerate(seqs):
        n = (s + BLOCK - 1) // BLOCK
        # physical pages are a permutation of the logical ones: production never maps
        # identity, and the planner must not care
        perm = torch.randperm(nb, generator=g)[:n]
        bt[r, :n] = perm.to(torch.int32)
    topk_idx = torch.zeros(KVH, total_q, topk, dtype=torch.int32)
    row = 0
    for r, (l, p) in enumerate(zip(lens, prefixes)):
        for i in range(l):
            v = (p + i + BLOCK) // BLOCK            # causal: blocks this row may select
            if mode == "empty" or v <= 0:
                picks = [0] * topk                   # all sentinels below
            elif mode == "same":                     # every row selects the same blocks
                picks = list(range(min(v, topk)))
            elif mode == "dup":                      # duplicates inside a row
                picks = [(i // 3 + j // 4) % max(v, 1) for j in range(topk)]
            else:
                k = min(topk, v)
                picks = torch.randperm(v, generator=g)[:k].tolist() + [0] * (topk - k)
            topk_idx[:, row, :len(picks)] = torch.tensor(picks, dtype=torch.int32)[None, :]
            row += 1
        if mode == "empty":
            topk_idx[:, sum(lens[:r]):sum(lens[:r + 1]), :] = -1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                      dtype=torch.int32, device=DEV)
    return pool, bt.to(DEV), topk_idx.to(DEV), cu


def run_case(ref, v2, name, nb, lens, prefixes, cap, seed=1, mode="normal", topk=TOPK):
    pool, bt, topk_idx, cu = make_case(ref, nb, lens, prefixes, seed, mode, topk)
    a = ref.plan_prefill_slices_fast(pool, bt, topk_idx, cu, KVH, cap)
    b = v2.plan_prefill_slices_v2(pool, bt, topk_idx, cu, KVH, cap)
    problems = []
    if len(a) != len(b):
        problems.append(f"n_slices {len(a)} != {len(b)}")
    else:
        for i, (x, y) in enumerate(zip(a, b)):
            if x["reqs"] != y["reqs"]:
                problems.append(f"slice {i}: reqs {x['reqs']} != {y['reqs']}")
            elif tuple(x["q"]) != tuple(y["q"]):
                problems.append(f"slice {i}: q {x['q']} != {y['q']}")
            elif x["n"] != y["n"]:
                problems.append(f"slice {i}: n {x['n']} != {y['n']}")
            elif not bool((x["pages"] == y["pages"]).all()):
                problems.append(f"slice {i}: page lists differ ({x['n']} ids)")
    check(f"{name} (nb={nb} cap={cap} rows={sum(lens)} slices={len(a)})",
          not problems, problems[0] if problems else "")
    del pool, bt, topk_idx
    if DEV == "cuda":
        torch.cuda.empty_cache()


# (name, nb, lens, prefixes, cap, kwargs, needs_gpu)
CASES = [
    ("empty selection (all -1)", 512, [256], [0], 384, dict(mode="empty"), False),
    ("empty selection, cap 1", 512, [64], [0], 1, dict(mode="empty"), False),
    ("single row", 512, [1], [0], 384, {}, False),
    ("single row, long prefix", 9042, [1], [262144 - 128], 384, {}, False),
    ("cap 1 (every row its own slice)", 512, [128], [0], 1, {}, False),
    ("cap 2", 512, [256], [0], 2, {}, False),
    ("cap == topk (one row exactly fills a slice)", 512, [256], [0], 16, {}, False),
    ("cap == topk - 1 (row alone exceeds the cap: hard floor)", 512, [256], [0], 15, {}, False),
    ("cap >= nb (no split possible)", 128, [4096], [0], 512, {}, False),
    ("nb smaller than topk (partial blocks at the causal edge)", 8, [64], [0], 4, {}, False),
    ("topk 1", 512, [512], [0], 8, dict(topk=1), False),
    ("topk 64 (union wider than the cap)", 2048, [1024], [0], 32, dict(topk=64), False),
    ("many tiny requests", 4096, [7] * 64, [65536] * 64, 384, {}, False),
    ("one long request, one short", 4096, [8000, 33], [32768, 0], 384, {}, False),
    ("prefix not a block multiple", 4096, [1000], [12345], 384, {}, False),
    ("rows == 2, cap forces a split mid-row-pair", 4096, [2], [262144], 8, {}, False),
    # the served geometry: nb = 9,042 pool pages per rank per layer, 16K-token chunks
    ("served pool, 16K chunk, prefix 0", 9042, [16384], [0], 384, {}, True),
    ("served pool, 16K chunk, 192K prefix, same rows", 9042, [16384], [196608], 384,
     dict(mode="same"), True),
    ("served pool, 16K chunk, 192K prefix, random", 9042, [16384], [196608], 384, {}, True),
    ("served pool, 16K chunk, 128K prefix, random", 9042, [16384], [131072], 384, {}, True),
    ("served pool, 8K chunk, 56K prefix, random", 9042, [8192], [57344], 384, {}, True),
    ("served pool, 16K chunk, 192K prefix, duplicate pages", 9042, [16384], [196608], 384,
     dict(mode="dup"), True),
    ("served pool, 6 requests at 48K prefix", 9042, [4096] * 6, [49152] * 6, 384, {}, True),
    ("served pool, 6 requests, identical rows", 9042, [2048] * 6, [98304] * 6, 384,
     dict(mode="same"), True),
    ("served pool, cap > nb, long prefix", 9042, [8192], [98304], 20000, {}, True),
]


def main() -> int:
    global DEV
    sys.dont_write_bytecode = True   # do not litter the staged tree with __pycache__
    want_all = "--all" in sys.argv
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    src = staged_src()
    check("patch 0020 staged: slice_planner.py in the build context",
          os.path.isfile(os.path.join(src, "slice_planner.py")), src)
    q4kv = open(os.path.join(src, "minimax_q4kv.py")).read()
    check("the call site carries the VLLM_EXL3_Q4KV_PLANNER_V2 switch",
          "VLLM_EXL3_Q4KV_PLANNER_V2" in q4kv and "plan_prefill_slices_v2" in q4kv)
    ref, v2 = load_planners(src)
    print(f"device={DEV}  cases={'all' if want_all else 'small'}")
    skipped = 0
    for name, nb, lens, prefixes, cap, kw, heavy in CASES:
        if heavy and not want_all:
            skipped += 1
            continue
        if heavy and DEV != "cuda":
            print(f"SKIP {name}: served-geometry cases need a GPU (pass --all inside the image)")
            skipped += 1
            continue
        run_case(ref, v2, name, nb, lens, prefixes, cap, **kw)
    if skipped:
        print(f"{skipped} served-geometry case(s) not run; use --all on a GPU host")
    for d in _STAGE:
        shutil.rmtree(d, ignore_errors=True)
    print("OK" if not FAILS else f"FAILED: {len(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

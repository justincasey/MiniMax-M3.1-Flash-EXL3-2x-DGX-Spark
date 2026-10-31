#!/usr/bin/env python3
"""Static pack-layout and harness unit tests (no GPU).

  python3 tests/test_pack_layout.py

Covers the serve-config merge, the 2.15 pack manifest and census, the fabric
plan, the manifest splitter/verifier, and the build's staging step: both pinned
engine commits fetched from GitHub (cached under .work/test-src) with both patch
series applied. That last part needs network access to github.com.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILS.append(name)
        print(f"FAIL {name}" + (f": {detail}" if detail else ""))


def load(path: str):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def test_census():
    census = load("config/lanes/2.15bpw/pack-census.json")
    modules = census["modules"]
    check("census repo", census["repo"] == "vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw", census["repo"])
    check("census revision", census["revision"] == "99bf3a57af5d07245ea9abb5426e7dce6af6356b")
    check("60 layers", all(f"language_model.model.layers.{i}.self_attn.q_proj" in modules for i in range(60)))
    check("57 MoE gates", sum(1 for k in modules if k.endswith(".block_sparse_moe.gate")) == 57)


# The 2.15 lane needs no overlay; the merge path is kept for lanes that do, and is tested
# with an MXFP8/EXL3-split overlay fixture.
OVERLAY = "tests/data/overlay-mxfp8-exl3.json"


def test_serve_config():
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "serve.json")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/make_serve_config.py"),
         "--base", os.path.join(ROOT, "tests/data/pack-config.json"),
         "--overlay", os.path.join(ROOT, OVERLAY),
         "--out", out], capture_output=True, text=True)
    check("serve config merges", r.returncode == 0, r.stderr)
    base = load("tests/data/pack-config.json")
    merged = load(os.path.relpath(out, ROOT)) if out.startswith(ROOT) else json.load(open(out))
    check("serve config preserves the model", set(base) == set(merged))
    check("serve config keeps 60 layers", merged["text_config"]["num_hidden_layers"] == 60)
    check("serve config keeps the model definition byte-identical",
          all(json.dumps(merged[k], sort_keys=True) == json.dumps(base[k], sort_keys=True)
              for k in base if k != "quantization_config"))
    check("serve config keeps the pack's own quant keys",
          all(merged["quantization_config"].get(k) == v for k, v in base["quantization_config"].items()))
    check("serve config replaces quant only",
          merged["quantization_config"] == load(OVERLAY)["quantization_config"])
    check("serve config is deterministic", subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/make_serve_config.py"),
         "--base", os.path.join(ROOT, "tests/data/pack-config.json"),
         "--overlay", os.path.join(ROOT, OVERLAY),
         "--check", out], capture_output=True, text=True).returncode == 0)
    doubled = dict(base)
    doubled["quantization_config"] = merged["quantization_config"]
    doubled_path = os.path.join(tmp, "doubled.json")
    with open(doubled_path, "w") as f:
        json.dump(doubled, f)
    check("serve config refuses a double overlay", subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/make_serve_config.py"),
         "--base", doubled_path, "--overlay", os.path.join(ROOT, OVERLAY)],
        capture_output=True, text=True).returncode != 0)


def test_manifest():
    rows = []
    with open(os.path.join(ROOT, "config/lanes/2.15bpw/pack-manifest.tsv")) as f:
        for line in f:
            if line.strip():
                rows.append(line.rstrip("\n").split("\t"))
    check("40 manifest rows", len(rows) == 40, str(len(rows)))
    check("manifest hashes", all(len(r[0]) == 64 and all(c in "0123456789abcdef" for c in r[0]) for r in rows))
    check("manifest total bytes", sum(int(r[1]) for r in rows) == 118985525140)
    check("16 shards", sum(1 for r in rows if r[2].startswith("model-") and r[2].endswith(".safetensors")) == 16)
    check("dspark present", any(r[2] == "dspark/model-dspark-00001.safetensors" for r in rows))
    check("no duplicate paths", len({r[2] for r in rows}) == len(rows))


def test_split_and_verify():
    tmp = tempfile.mkdtemp()
    files = []
    for i, size in enumerate([100, 200, 300, 400, 500]):
        path = os.path.join(tmp, f"f{i}.bin")
        with open(path, "wb") as f:
            f.write(bytes([i]) * size)
        files.append((path, size))
    manifest = os.path.join(tmp, "manifest.tsv")
    import hashlib
    with open(manifest, "w") as f:
        for path, _ in files:
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            f.write(f"{digest}\t{os.path.getsize(path)}\t{os.path.basename(path)}\n")

    parts_dir = os.path.join(tmp, "parts")
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/split_manifest.py"), "--manifest", manifest,
         "--parts", "2", "--out-dir", parts_dir],
        capture_output=True, text=True,
    )
    check("split runs", out.returncode == 0, out.stderr)
    part0 = open(os.path.join(parts_dir, "part0.txt")).read().split()
    part1 = open(os.path.join(parts_dir, "part1.txt")).read().split()
    check("split covers every file", sorted(part0 + part1) == sorted(r[2] for r in rows_for(manifest)))
    sizes = {os.path.basename(p): os.path.getsize(p) for p, _ in files}
    delta = abs(sum(sizes[p] for p in part0) - sum(sizes[p] for p in part1))
    check("split is balanced", delta <= max(sizes.values()), f"delta {delta}")

    ok = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/verify_manifest.py"), "--pack", tmp,
         "--manifest", manifest], capture_output=True, text=True,
    )
    check("verify passes", ok.returncode == 0, ok.stderr or ok.stdout)
    with open(os.path.join(tmp, "f3.bin"), "r+b") as f:
        f.write(b"x")
    bad = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/verify_manifest.py"), "--pack", tmp,
         "--manifest", manifest], capture_output=True, text=True,
    )
    check("verify catches a hash mismatch", bad.returncode == 1)
    quick = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/verify_manifest.py"), "--pack", tmp,
         "--manifest", manifest, "--quick"], capture_output=True, text=True,
    )
    check("quick verify misses a same-size corruption", quick.returncode == 0)


def rows_for(manifest: str):
    for line in open(manifest):
        if line.strip():
            _, _, path = line.rstrip("\n").split("\t")
            yield (None, None, path)


def test_fabric():
    plan = load("tests/data/fabric-plan.json")
    check("gid", plan["gid"] == 3)
    check("subnets", plan["subnets"] == ["10.0.226.0/24", "10.0.227.0/24"], str(plan["subnets"]))
    check("rank0 ip", plan["rank"][0]["ip"] == "10.0.226.2")
    check("rank1 ip", plan["rank"][1]["ip"] == "10.0.226.1")
    check("both cable functions", plan["rank"][0]["hcas"] == ["rocep1s0f1", "roceP2p1s0f1"])
    check("uniform MTU", plan["mtu_uniform"] is True)


def pins() -> dict:
    keys = ("PATCH_FILES", "EXL3_PATCH_FILES", "EXL3_REPO", "EXL3_REV", "PLUGIN_REPO", "PLUGIN_REV")
    out = subprocess.run(
        ["bash", "-c", '. config/pins.sh >/dev/null && printf "%s\\n" ' + " ".join(f'"${k}"' for k in keys)],
        cwd=ROOT, env={**os.environ, "ROOT": ROOT, "LANE": "2.15bpw"}, capture_output=True, text=True, check=True)
    return dict(zip(keys, out.stdout.split("\n")))


def run_prepare_src(p: dict, build: str, **overrides) -> subprocess.CompletedProcess:
    """Run the build's staging script exactly as `m31.sh build` does, minus docker."""
    import shutil
    ctx = os.path.join(build, "docker")
    os.makedirs(ctx, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "docker", "Dockerfile"), ctx)
    env = {**os.environ, "BASE_IMAGE_DIGEST": "test",
           "EXL3_REPO": p["EXL3_REPO"], "EXL3_REV": p["EXL3_REV"],
           "PLUGIN_REPO": p["PLUGIN_REPO"], "PLUGIN_REV": p["PLUGIN_REV"],
           "EXL3_PATCHES": " ".join(os.path.join(ROOT, x) for x in p["EXL3_PATCH_FILES"].split()),
           **overrides}
    # The source cache persists across runs so only the first run downloads.
    src = os.path.join(ROOT, ".work", "test-src")
    return subprocess.run(
        ["bash", os.path.join(ROOT, "scripts/prepare_src.sh"), src, build]
        + [os.path.join(ROOT, x) for x in p["PATCH_FILES"].split()],
        env=env, capture_output=True, text=True)


# Digests of the two engine trees in the build context that was validated on hardware (both patch series applied).
# A pin or patch change is a release: rebuild, re-validate, then update these.
VALIDATED_TREES = {
    "exllamav3-src": "06d5af1de8eb66454c17298e905d5cd1aee68961929a474ed42ff993d53ca132",
    "vllm-exl3-src": "4a2bc6c875a33d10e750d90c918a6a02f7f4f48957fa784274a9c555abecb31a",
}


def tree_digest(root: str) -> str:
    """sha256 over every file's content hash, executable bit and path (umask-independent)."""
    import hashlib
    rows = []
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            x = "x" if os.stat(p).st_mode & 0o100 else "-"
            rows.append(f"{h} {x} {os.path.relpath(p, root)}\n".encode())
    return hashlib.sha256(b"".join(sorted(rows))).hexdigest()


def test_prepare_src():
    p = pins()
    with tempfile.TemporaryDirectory() as tmp:
        out = run_prepare_src(p, os.path.join(tmp, "build"))
        check("prepare_src fetches the pinned commits and stages the build context",
              out.returncode == 0, out.stderr[-400:])
        if out.returncode != 0:
            return
        ctx = os.path.join(tmp, "build", "docker")
        plugin = os.path.join(ctx, "vllm-exl3-src")
        exl3 = os.path.join(ctx, "exllamav3-src")
        text = open(os.path.join(plugin, "src/vllm_exl3/exl3.py")).read()
        check("plugin series carries every fix", "_lm_head_full_vocab" in text and "_indexer_rows_for_rank" in text
              and 'getattr(layer, "use_ep", None) is False' in text
              and "elif moe_act == MOE_ACT_SWIGLU_OAI:" in text and "def _narrow_lm_head" in text
              and "FAT_THRESHOLD_PREFILL" in text and "exl3_moe_coop_mk" in text
              and os.path.exists(os.path.join(plugin, "src/vllm_exl3/weight_snapshot.py")))
        bindings = open(os.path.join(exl3, "exllamav3/exllamav3_ext/bindings.cpp")).read()
        check("exllamav3 series carries the coop and dense-decode kernels",
              "exl3_moe_coop_mk" in bindings
              and os.path.exists(os.path.join(exl3, "exllamav3/exllamav3_ext/quant/exl3_dgemv.cu")))
        check("context keeps the upstream license files",
              os.path.exists(os.path.join(plugin, "THIRD_PARTY_NOTICES.md"))
              and os.path.exists(os.path.join(exl3, "LICENSE")))
        check("no .git in the context", not any(".git" in d for _, d, _ in os.walk(ctx)))
        for name, want in VALIDATED_TREES.items():
            got = tree_digest(os.path.join(ctx, name))
            check(f"staged {name} matches the hardware-validated tree", got == want, got)
        pins_txt = open(os.path.join(ctx, "PINS")).read()
        check("PINS records both sources at their revs and the series hash",
              f"exllamav3_rev={p['EXL3_REV']}" in pins_txt and f"exllamav3_source={p['EXL3_REPO']}" in pins_txt
              and f"vllm_exl3_rev={p['PLUGIN_REV']}" in pins_txt
              and f"vllm_exl3_source={p['PLUGIN_REPO']}" in pins_txt
              and "patch_series_sha256=" in pins_txt)
        bogus = "0" * 40
        out = run_prepare_src(p, os.path.join(tmp, "build-bogus"), PLUGIN_REV=bogus)
        check("prepare_src refuses a commit the repo does not have",
              out.returncode != 0 and f"cannot fetch vllm-exl3 {bogus}" in out.stderr, out.stderr[-300:])


def main() -> int:
    test_census()
    test_serve_config()
    test_manifest()
    test_split_and_verify()
    test_fabric()
    test_prepare_src()
    print("OK" if not FAILS else f"FAILED: {len(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

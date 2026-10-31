#!/usr/bin/env bash
# Static contract tests: no GPU, no ssh, no network.
#
#   bash tests/test_recipe.sh
#
# Generates both ranks' guard + docker + vllm serve commands from pinned
# fabric fixtures and checks the invariants a deployment depends on, then the
# pack manifest, overlay, census, pins and the invalid-config gates.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

fails=0
ok() { printf 'PASS %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }
has() { grep -qF -- "$2" <<<"$1"; }
lacks() { ! grep -qF -- "$2" <<<"$1"; }
unesc() { sed 's/\\//g'; }

# Hermetic: no caller env file, lane or profile leaks into the defaults under test.
unset LANE PROFILE REFERENCE MAX_MODEL_LEN MAX_NUM_SEQS KVB CAPTURE_SIZES EXTRA_DOCKER_ARGS EXTRA_VLLM_ARGS
export ENV_FILE=/dev/null
export NODES="192.0.2.11 192.0.2.12"
export FABRIC_PLAN="$ROOT/tests/data/fabric-plan.json"
export BASE_DIR=/home/spark/m31
export RUN_DIR=/tmp/m31-test-run

rank_cmd() {  # rank_cmd RANK -> unescaped docker line
  ./m31.sh print-serve "$1" 2>/dev/null | sed -n 2p | unesc
}

# ── syntax ──────────────────────────────────────────────────────────────────
shopt -s nullglob
for f in m31.sh scripts/*.sh tests/*.sh; do
  bash -n "$f" && ok "shell syntax $f" || bad "shell syntax $f"
done
python3 -m py_compile bench/*.py scripts/*.py tests/*.py \
  && ok "python syntax" || bad "python syntax"
shopt -u nullglob

# remote() passes variables by name in the prelude and no positional args:
# a remote script that reads $1..$8 dies under set -u (and would silently
# blank a path without it). Catch that class before it reaches a node.
if grep -nE '^[[:space:]]+[A-Z_]+=\$[0-9]|cd "\$[0-9]"' m31.sh >/tmp/m31-pos.txt; then
  bad "remote script uses positional args: $(tr '\n' ' ' </tmp/m31-pos.txt)"
else
  ok "remote scripts use named variables"
fi

# remote() feeds its script to `bash -s` on stdin: an inner ssh that is not
# fed by a pipe (or run by rsync -e) must use -n, or it swallows the rest of
# the script (a copy once "finished" 111 GiB in 6 s that way).
if grep -nE 'ssh -o BatchMode' m31.sh | grep -vE '\| ssh -o BatchMode|-e "ssh -o BatchMode' >/tmp/m31-ssh.txt; then
  bad "nested ssh without -n: $(tr '\n' ' ' </tmp/m31-ssh.txt)"
else
  ok "nested ssh is -n or pipe-fed"
fi

# ── rank 0: the served rank ─────────────────────────────────────────────────
# The emitted line is run by a remote shell: it must split into the docker argv,
# with the JSON flags intact as single words (unesc above hides quoting bugs).
line=$(./m31.sh print-serve 0 2>/dev/null | sed -n 2p)
eval "set -- $line"
if [[ "${1:-}" == docker && "${2:-}" == run && $# -gt 80 ]] && printf '%s\n' "$@" | grep -qx -- '--speculative-config'; then
  ok "docker line splits into $# words"
else
  bad "docker line does not split into argv (first word: ${1:0:40})"
fi
printf '%s\n' "$@" | grep -q '^{"method":"dspark",' && ok "spec config is one word" || bad "spec config split"
eval "set -- $(EXTRA_DOCKER_ARGS='--label x=y' ./m31.sh print-serve 0 2>/dev/null | sed -n 2p)"
args=" $* "; [[ "${args%% --entrypoint *}" == *"--label x=y"* ]] && ok "EXTRA_DOCKER_ARGS precede the image" || bad "EXTRA_DOCKER_ARGS after the image"
c=$(rank_cmd 0) || bad "print-serve 0 failed"
for want in \
  "docker run -d --name m31-flash-tp2" \
  "--network host" "--ipc host" "--gpus all" "--device /dev/infiniband" \
  "--cap-add IPC_LOCK" "--ulimit memlock=-1:-1" "--ulimit core=0" "--shm-size 2g" \
  "--memory=118g" "--memory-swap=118g" "--oom-score-adj 1000" "--security-opt seccomp=unconfined" \
  "-v /home/spark/models:/models:ro" "-v /home/spark/.cache/m31:/cache:rw" \
  "-v /home/spark/m31/serve-config.json:/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw/config.json:ro" \
  "--entrypoint vllm" "serve /models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw" \
  "--tensor-parallel-size 2" "--nnodes 2" "--node-rank 0" \
  "--master-addr 10.0.226.2" "--master-port 29511" \
  "--served-model-name minimax-m3.1-flash" "--host 0.0.0.0" "--port 8000" \
  "--language-model-only" "--block-size 128" "--disable-hybrid-kv-cache-manager" \
  "--max-model-len 300000" "--max-num-seqs 6" "--max-num-batched-tokens 16384" \
  "--kv-cache-memory-bytes 20000000000" \
  "--gpu-memory-utilization 0.85" "--kv-cache-dtype nvfp4" \
  "indexer_kv_dtype" "fp8" "method" "dspark" "num_speculative_tokens" \
  "TRITON_ATTN" "draft_sample_method" "probabilistic" "rejection_sample_method" "block" \
  "FULL_DECODE_ONLY" "cudagraph_capture_sizes" "max-cudagraph-capture-size 24" \
  "reasoning-parser minimax_m3" "tool-call-parser minimax_m3" "--enable-auto-tool-choice" \
  "VLLM_EXL3_Q4KV=1" "VLLM_EXL3_INDEXER_MXFP8=1" "MIMALLOC_PURGE_DELAY=0" "MIMALLOC_RESET_DELAY=0" \
  "VLLM_EXL3_TRIM_AFTER_LAYER=1" "VLLM_EXL3_UMA_POSTLOAD=1" "VLLM_EXL3_ATS_ALIAS=1" \
  "VLLM_EXL3_DRAFT_VOCAB_META=1" "VLLM_EXL3_TRIM_BEFORE_KV=1" "VLLM_EXL3_MOE_TEMP_ROWS=16384" \
  "VLLM_EXL3_FAT_THRESHOLD=128" "VLLM_EXL3_FAT_THRESHOLD_PREFILL=0" "VLLM_EXL3_FAT_PREFILL_TOKENS=1024" \
  "VLLM_HOST_IP=10.0.226.2" "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1" "NCCL_IB_GID_INDEX=3" \
  "NCCL_SOCKET_IFNAME=enp1s0f1np1" "GLOO_SOCKET_IFNAME=enp1s0f1np1" \
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" \
  "--override-generation-config"; do
  has "$c" "$want" && ok "rank0 has: $want" || bad "rank0 missing: $want"
done
lacks "$c" "--headless" && ok "rank 0 is not headless" || bad "rank 0 must serve the API"
lacks "$c" "--api-key" && ok "no --api-key in argv" || bad "API key must not be in argv"
lacks "$c" "gpu-memory-utilization 0.92" && ok "no default GMU" || bad "GMU leftover"

# ── rank 1: headless ────────────────────────────────────────────────────────
c1=$(rank_cmd 1) || bad "print-serve 1 failed"
has "$c1" "--headless" && ok "rank 1 is headless" || bad "rank 1 missing --headless"
has "$c1" "--node-rank 1" && ok "rank 1 node-rank" || bad "rank 1 node-rank"
has "$c1" "VLLM_HOST_IP=10.0.226.1" && ok "rank 1 VLLM_HOST_IP" || bad "rank 1 VLLM_HOST_IP"
lacks "$c1" "--served-model-name" && ok "rank 1 has no served-model-name" || bad "rank 1 must not pass served-model-name"
lacks "$c1" "--host" && ok "rank 1 has no host" || bad "rank 1 must not pass --host"
lacks "$c1" "--port" && ok "rank 1 has no port" || bad "rank 1 must not pass --port"

# ── guard line ──────────────────────────────────────────────────────────────
g=$(./m31.sh print-serve 0 2>/dev/null | sed -n 1p)
for want in "memguard.sh watch m31-flash-tp2" "2097152" "1048576" "65536" "20"; do
  has "$g" "$want" && ok "guard has: $want" || bad "guard missing: $want"
done

# ── API key: env on rank 0 only ─────────────────────────────────────────────
c0k=$(API_KEY=secret ./m31.sh print-serve 0 2>/dev/null | sed -n 2p)
c1k=$(API_KEY=secret ./m31.sh print-serve 1 2>/dev/null | sed -n 2p)
has "$c0k" "VLLM_API_KEY=secret" && ok "API key reaches rank 0 env" || bad "API key missing on rank 0"
lacks "$c1k" "VLLM_API_KEY" && ok "API key absent on rank 1" || bad "API key leaked to rank 1"

# ── profiles ────────────────────────────────────────────────────────────────
# 2.15-lane context profiles: pinned pools (14 / 28 GiB), shared prefill knobs
c=$(PROFILE=q4-215-256k-2s ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
has "$c" "--max-model-len 262144" && has "$c" "--max-num-seqs 2" && has "$c" "--kv-cache-memory-bytes 15032385536" \
  && has "$c" "--max-num-batched-tokens 16384" && has "$c" "VLLM_EXL3_FAT_THRESHOLD_PREFILL=0" \
  && ok "q4-215-256k-2s profile (14 GiB pool, 2.15 prefill knobs)" || bad "q4-215-256k-2s profile"
c=$(PROFILE=q4-215-512k-8s ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
has "$c" "--max-model-len 524288" && has "$c" "--max-num-seqs 8" && has "$c" "--kv-cache-memory-bytes 30064771072" \
  && has "$c" "max-cudagraph-capture-size 32" && ok "q4-215-512k-8s profile (28 GiB cap, capture to 32)" || bad "q4-215-512k-8s profile"

# caller env wins over the profile
c=$(MAX_MODEL_LEN=4096 ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
has "$c" "--max-model-len 4096" && ok "caller env overrides the profile" || bad "caller env did not override the profile"

# explicit KVB wins over the derivation
c=$(KVB=123456789 ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
has "$c" "--kv-cache-memory-bytes 123456789" && ok "explicit KVB" || bad "explicit KVB"

# derived KVB (a profile that sets no KVB; every shipped profile pins it):
# 65536 x 2 x 28160 x 1.02 aligned to 256 MiB
tmp_profile=config/profiles/zz-contract-derived-kvb.env
trap 'rm -f "$tmp_profile"' EXIT
printf ': "${MAX_MODEL_LEN:=65536}"\n: "${MAX_NUM_SEQS:=2}"\n' > "$tmp_profile"
c=$(PROFILE=zz-contract-derived-kvb ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
rm -f "$tmp_profile"
has "$c" "--kv-cache-memory-bytes 4026531840" && ok "derived KVB (3.75 GiB)" || bad "derived KVB: $(grep -o 'kv-cache-memory-bytes [0-9]*' <<<"$c")"

# ── toggles ─────────────────────────────────────────────────────────────────
c=$(DSPARK=0 KV_Q4=0 KV_CACHE_DTYPE=fp8 ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc || true)
lacks "$c" "--speculative-config" && has "$c" "max-cudagraph-capture-size 2" \
  && ok "DSPARK=0 drops speculation and captures one row per stream" || bad "DSPARK=0"
has "$c" "VLLM_EXL3_Q4KV=0" && ok "KV_Q4=0 reaches the container" || bad "KV_Q4=0 not forwarded"
c=$(INDEXER_MXFP8=0 ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
has "$c" "VLLM_EXL3_INDEXER_MXFP8=0" && ok "INDEXER_MXFP8=0 reaches the container" || bad "INDEXER_MXFP8=0 not forwarded"
c=$(EAGER=1 ./m31.sh print-serve 0 2>/dev/null | sed -n 2p | unesc)
has "$c" "--enforce-eager" && lacks "$c" "--compilation-config" && ok "EAGER=1" || bad "EAGER=1"

# ── fail-closed config gates (no output on failure) ─────────────────────────
expect_fail() {  # expect_fail "label" ENV=VAL...
  local label=$1; shift
  local out rc=0
  out=$(env "$@" ./m31.sh print-serve 0 2>/dev/null) || rc=$?
  if (( rc != 0 )) && [[ -z "$out" ]]; then ok "rejected: $label"; else bad "$label was accepted (rc=$rc, output ${#out} bytes)"; fi
}
expect_fail "KV_Q4 with fp8 main KV" KV_Q4=1 KV_CACHE_DTYPE=fp8
expect_fail "KV_Q4 with bf16 indexer" KV_Q4=1 INDEXER_KV_DTYPE=bf16
expect_fail "KV_Q4 with baseline draft attn" KV_Q4=1 DRAFT_ATTN_BACKEND=FLASHINFER
expect_fail "nvfp4 without KV_Q4" KV_Q4=0 KV_CACHE_DTYPE=nvfp4
expect_fail "batched tokens not a multiple of 128" MAX_NUM_BATCHED_TOKENS=4000
expect_fail "zero speculative tokens" DSPARK=1 NUM_SPECULATIVE_TOKENS=0
expect_fail "missing profile" PROFILE=nope
expect_fail "unknown lane" LANE=nope
expect_fail "partial fabric pins" RANK0_IP=10.0.0.1

# ── pins, manifest, census, overlay ─────────────────────────────────────────
grep -q '^BASE_IMAGE_DIGEST=.*86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c' config/pins.sh \
  && ok "base image digest pin" || bad "base image digest pin"
grep -q '^PLUGIN_REV=.*053c634db54afc981becfb46c0482e7358cc00ba' config/pins.sh \
  && ok "plugin rev pin" || bad "plugin rev pin"
grep -q '^EXL3_REV=.*fdd77361971144e8d175e09e7b50b7100739fd95' config/pins.sh \
  && ok "exllamav3 rev pin" || bad "exllamav3 rev pin"

# The build fetches exactly the pinned commit, so an abbreviated rev is refused before anything runs.
out=$(EXL3_REV=fdd77361 ./m31.sh print-serve 0 2>&1 || true)
grep -q "must be full 40-character commit shas" <<<"$out" && ok "abbreviated engine rev refused" \
  || bad "abbreviated engine rev accepted: ${out:0:120}"

python3 scripts/fabric_plan.py --rank0 tests/data/fabric-head.json --rank1 tests/data/fabric-worker.json -o /tmp/m31-plan.json >/dev/null \
  && python3 - <<'PY' && ok "fabric plan" || bad "fabric plan"
import json
p = json.load(open("/tmp/m31-plan.json"))
assert p["gid"] == 3, p
assert p["subnets"] == ["10.0.226.0/24", "10.0.227.0/24"], p["subnets"]
assert p["rank"][0]["ip"] == "10.0.226.2" and p["rank"][1]["ip"] == "10.0.226.1"
assert p["rank"][0]["hcas"] == ["rocep1s0f1", "roceP2p1s0f1"]
assert p["rank"][1]["hcas"] == ["rocep1s0f1", "roceP2p1s0f1"]
assert p["rendezvous"] == "10.0.226.2"
assert p["mtu_uniform"] is True
PY

# The plugin patches must be valid mailbox patches; 0002 is tonyd2wild's file byte for byte.
grep -q '^Subject: \[PATCH\]' patches/0001-minimax-exl3-lmhead-tp2.patch \
  && grep -q '_lm_head_full_vocab' patches/0001-minimax-exl3-lmhead-tp2.patch \
  && ok "LM-head TP2 patch" || bad "LM-head TP2 patch"
python3 - <<'PY2' && ok "indexer TP2 patch is tonyd2wild's 70c4e43 file" || bad "indexer TP2 patch"
import hashlib
assert hashlib.sha256(open("patches/0002-minimax-exl3-indexer-tp2.patch", "rb").read()).hexdigest() \
    == "8af3e8600f0ef9856a93b77ce4d6ba5e520ac8c58a456d66b6b94f68c89b10ac"
PY2

# ── lane 2.15bpw ────────────────────────────────────────────────────────────
python3 - <<'PY' && ok "2.15 manifest" || bad "2.15 manifest"
rows = [l.rstrip("\n").split("\t") for l in open("config/lanes/2.15bpw/pack-manifest.tsv") if l.strip()]
assert len(rows) == 40, len(rows)
assert all(len(r) == 3 and len(r[0]) == 64 for r in rows)
assert sum(int(r[1]) for r in rows) == 118985525140
assert any(r[2] == "dspark/model-dspark-00001.safetensors" for r in rows)
PY
c=$(rank_cmd 0)
miss=0
for want in "serve /models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw" \
            "-v /home/spark/m31/serve-config.json:/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw/config.json:ro" \
            "--max-model-len 300000" "--max-num-seqs 6" "--kv-cache-memory-bytes 20000000000" \
            'cudagraph_capture_sizes":[1,2,3,4,8,12,16,24]' "--max-cudagraph-capture-size 24"; do
  has "$c" "$want" || { bad "2.15 lane: missing $want"; miss=1; }
done
(( miss == 0 )) && ok "2.15 lane serves the q4-300k flag set"
printf 'LANE=2.15bpw\n' > /tmp/m31-lane.env
c=$(ENV_FILE=/tmp/m31-lane.env rank_cmd 0)
has "$c" "serve /models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw" && ok "LANE is read from ENV_FILE" || bad "LANE from ENV_FILE"
out=$(LANE=nope ENV_FILE=/tmp/m31-lane.env ./m31.sh print-serve 0 2>&1 || true)
grep -q "LANE=nope: no such lane" <<<"$out" && ok "caller LANE beats ENV_FILE" || bad "caller LANE precedence"

# Canonical Apache-2.0 text (an abridged copy truncates sections 8 and 9).
python3 - <<'PY2' && ok "LICENSE is canonical Apache-2.0" || bad "LICENSE is not the canonical Apache-2.0 text"
import hashlib
assert hashlib.sha256(open("LICENSE", "rb").read()).hexdigest() == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
PY2

# Every ./m31.sh <command> in the docs must be a dispatcher case.
python3 - <<'PY2' && ok "documented commands exist" || bad "documented commands missing from the dispatcher"
import re, sys
src = open("m31.sh").read()
cases = set()
for block in re.findall(r"^\s*([a-z|-]+)\)", src.split("case ${1:-help} in", 1)[1], re.M):
    cases.update(block.split("|"))
missing = set()
for doc in ("README.md", "bench/README.md", "docs/FIRST-RUN.md", "AGENTS.md"):
    for cmd in re.findall(r"\./m31\.sh ([a-z][a-z-]*)", open(doc).read()):
        if cmd not in cases:
            missing.add((doc, cmd))
if missing:
    print(sorted(missing), file=sys.stderr)
    sys.exit(1)
PY2

(( fails == 0 )) && echo "OK" || { echo "FAILED: $fails"; exit 1; }

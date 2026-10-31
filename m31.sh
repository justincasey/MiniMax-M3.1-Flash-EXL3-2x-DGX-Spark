#!/usr/bin/env bash
# MiniMax-M3.1-Flash EXL3 SAGE 2.15bpw on two DGX Sparks (GB10, TP=2),
# behind an OpenAI-compatible vLLM endpoint.
#
#   ./m31.sh doctor        read-only checks on both nodes, finds the QSFP link
#   ./m31.sh fetch         pinned pack, split download, cable fan-out, sha256
#   ./m31.sh build         engine image on rank 0, shipped over the cable
#   ./m31.sh start         memory guards armed, rank 1 then rank 0; waits for /v1/models
#   ./m31.sh smoke         chat, tool call, reasoning separation, mixed-K mode
#   ./m31.sh probe         capability + metrics probe before trusting a benchmark
#   ./m31.sh bench [decode|prefill|needle|spec|pin|check|all]
#   ./m31.sh quality | longctx | tools | api-test
#   ./m31.sh stop | restart | status | logs [0|1] [n] | print-serve RANK
#
# Configuration: caller environment > .env > m31.sh defaults > config/profiles/.
# Release pins (config/pins.sh) are not tunable from .env.
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "m31.sh needs bash 4 or newer (this is $BASH_VERSION). On macOS: brew install bash" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── configuration ───────────────────────────────────────────────────────────
declare -A CALLER_ENV=()
while IFS= read -r key; do CALLER_ENV[$key]="${!key}"; done < <(compgen -e)

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
# LANE selects the pinned pack, so it is resolved before the pins: caller
# environment first, then a LANE= line in the env file.
if [[ -z "${LANE:-}" && -f "$ENV_FILE" ]]; then
  LANE=$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?LANE=[\"']?([A-Za-z0-9._-]+).*/\\2/p" "$ENV_FILE" | tail -1)
  [[ -n "$LANE" ]] || unset LANE
fi

# Pins first: .env can neither see nor override them.
# shellcheck source=config/pins.sh
. "$ROOT/config/pins.sh"

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[2]} value=${BASH_REMATCH[3]}
    if [[ "$value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      value=${BASH_REMATCH[1]}
    else
      value=${value%%#*}; value="${value%"${value##*[![:space:]]}"}"
    fi
    [[ -n "${!key+set}" ]] || export "$key=$value"
  done < "$ENV_FILE"
fi
for key in "${!CALLER_ENV[@]}"; do printf -v "$key" '%s' "${CALLER_ENV[$key]}"; export "$key"; done

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FAIL: $*"; exit 1; }
is_true() { case ${1:-} in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }

NODES="${NODES:-}"
SSH_USER="${SSH_USER:-spark}"
BASE_DIR="${BASE_DIR:-}"
MODELS_DIR="${MODELS_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
BUILD_DIR="${BUILD_DIR:-}"
CONTAINER="${CONTAINER:-m31-flash-tp2}"
MASTER_PORT="${MASTER_PORT:-29511}"

SERVED_NAME="${SERVED_NAME:-minimax-m3.1-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-}"
PROFILE="${PROFILE:-$LANE_DEFAULT_PROFILE}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"   # empty: the profile's value, else 4096
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-28160}"
KVB="${KVB:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-fp8}"
KV_Q4="${KV_Q4:-1}"
INDEXER_MXFP8="${INDEXER_MXFP8:-1}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-32768}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
WARMUP="${WARMUP:-1}"

DSPARK="${DSPARK:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_ATTN_BACKEND="${DRAFT_ATTN_BACKEND:-TRITON_ATTN}"
DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
DRAFT_SAMPLE="${DRAFT_SAMPLE:-probabilistic}"
REJECTION_SAMPLE="${REJECTION_SAMPLE:-block}"
EAGER="${EAGER:-0}"
CAPTURE_SIZES="${CAPTURE_SIZES:-}"
COMPILE_MODE="${COMPILE_MODE:-0}"

CONTAINER_MEMORY="${CONTAINER_MEMORY:-118g}"
GUARD_START_MIB="${GUARD_START_MIB:-2048}"
GUARD_MIB="${GUARD_MIB:-1024}"
GUARD_SWAP_MAX_MIB="${GUARD_SWAP_MAX_MIB:-64}"
DROP_CACHES_EVERY_S="${DROP_CACHES_EVERY_S:-20}"
DROP_CACHES_LOADING="${DROP_CACHES_LOADING:-1}"   # 0: the guard drops caches only once ready
PREFETCH="${PREFETCH:-0}"                         # 1: stream the shards into the page cache during container start
PREFETCH_PARALLEL="${PREFETCH_PARALLEL:-4}"
DROP_CACHES_START="${DROP_CACHES_START:-0}"       # 1: drop the page cache at start (weights restore with O_DIRECT, so a drop only evicts the image's libraries)
LAUNCH_STAGGER_S="${LAUNCH_STAGGER_S:-0}"         # seconds between the rank 1 and rank 0 launches; 0 = in parallel
WEIGHT_SNAPSHOT="${WEIGHT_SNAPSHOT:-1}"           # 1: per-rank snapshots of the loaded weights under CACHE_DIR (patches/0004)
CPUSET="${CPUSET:-}"
NCCL_DEBUG="${NCCL_DEBUG:-}"

RANK0_IP="${RANK0_IP:-}"; RANK1_IP="${RANK1_IP:-}"
RANK0_HCA="${RANK0_HCA:-}"; RANK1_HCA="${RANK1_HCA:-}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"
FABRIC_IF_0="${FABRIC_IF_0:-}"; FABRIC_IF_1="${FABRIC_IF_1:-}"

ALLOWED_CONTAINERS="${ALLOWED_CONTAINERS:-}"
STOP_OTHER="${STOP_OTHER:-}"; RESTORE_OTHER="${RESTORE_OTHER:-}"
REFERENCE="${REFERENCE:-}"   # token pin file (repo-relative); empty = bench/reference.json
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_BIN="${HF_BIN:-}"

# Profile: only fills values the caller/.env left unset (profiles use :=).
PROFILE_FILE="$ROOT/config/profiles/$PROFILE.env"
[[ -f "$PROFILE_FILE" ]] || die "PROFILE=$PROFILE: no such profile at $PROFILE_FILE"
# shellcheck source=/dev/null
. "$PROFILE_FILE"
PROFILE_MEASURED="${PROFILE_MEASURED:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# MoE prefill chunk (rows per fused launch). Chunks at least this large let
# experts with > FAT_THRESHOLD rows take the grouped dequant+GEMM tier.
MOE_TEMP_ROWS="${MOE_TEMP_ROWS:-$MAX_NUM_BATCHED_TOKENS}"
# Rows per expert above which an expert takes the fat tier (plugin default 256), and the
# smallest batch (tokens) that may use the fat tier at all (0: any).
FAT_THRESHOLD="${FAT_THRESHOLD:-256}"
FAT_THRESHOLD_PREFILL="${FAT_THRESHOLD_PREFILL:-$FAT_THRESHOLD}"
FAT_PREFILL_TOKENS="${FAT_PREFILL_TOKENS:-1024}"

RANK_HOSTS=()
if [[ -n "$NODES" ]]; then
  read -r -a RANK_HOSTS <<<"$NODES"
  (( ${#RANK_HOSTS[@]} == 2 )) || die "NODES must list exactly two targets, got: $NODES"
fi

BASE_DIR="${BASE_DIR:-/home/$SSH_USER/m31}"
MODELS_DIR="${MODELS_DIR:-/home/$SSH_USER/models}"
CACHE_DIR="${CACHE_DIR:-/home/$SSH_USER/.cache/m31}"
BUILD_DIR="${BUILD_DIR:-/home/$SSH_USER/build/m31}"
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
read -r -a PATCH_LIST <<<"$PATCH_FILES"
read -r -a EXL3_PATCH_LIST <<<"${EXL3_PATCH_FILES:-}"
# One sha256 over both series (plugin, then exllamav3), in apply order: the image tag and label
# carry it. An empty exllamav3 series leaves the hash of the plugin series unchanged.
PATCH_SHA256="$( (cd "$ROOT" && cat "${PATCH_LIST[@]}" ${EXL3_PATCH_LIST[@]+"${EXL3_PATCH_LIST[@]}"}) | { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; } | awk '{print $1}')"
IMAGE="${IMAGE:-m31-flash-exl3-tp2:$(printf '%s' "$PLUGIN_REV" | cut -c1-8)-$(printf '%s' "$PATCH_SHA256" | cut -c1-8)}"
URL="http://${RANK_HOSTS[0]:-localhost}:$PORT"
TOOLS="$BASE_DIR/tools"
RUN_DIR="${RUN_DIR:-$ROOT/runs/$(date +%Y%m%d-%H%M%S)}"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

# ── helpers ─────────────────────────────────────────────────────────────────
host_of() { [[ $1 == 0 ]] && echo "${RANK_HOSTS[0]}" || echo "${RANK_HOSTS[1]}"; }
target_of() { echo "$SSH_USER@$(host_of "$1")"; }

remote() {  # remote RANK SCRIPT [VAR=value ...]
  local rank=$1 script=$2; shift 2
  local prelude="set -euo pipefail"$'\n' kv
  for kv in "$@"; do prelude+="$(printf '%q' "${kv%%=*}")=$(printf '%q' "${kv#*=}")"$'\n'; done
  ssh "${SSH_OPTS[@]}" "$(target_of "$rank")" bash -s <<<"$prelude$script"
}

both() {  # both SCRIPT [VAR=value ...]
  local script=$1; shift
  local pids=() r rc=0
  for r in 0 1; do
    ( remote "$r" "$script" "$@" 2>&1 | sed -u "s/^/[rank$r] /"; exit "${PIPESTATUS[0]}" ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return "$rc"
}

rsync_to() {  # rsync_to RANK SRC DEST
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$2" "$(target_of "$1"):$3"
}

quoted() { local a out=""; while IFS= read -r a; do out+=" $(printf '%q' "$a")"; done; echo "${out# }"; }

jsonq() {  # jsonq FILE dotted.path -> scalar or comma-joined list
  python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    d = d[int(key)] if key.isdigit() else d[key]
print(",".join(str(x) for x in d) if isinstance(d, list) else d)
PY
}

need_hosts() {
  [[ -n "$NODES" ]] || die "set NODES in .env (two ssh targets, rank 0 first) — see .env.example"
  [[ -n "${RANK_HOSTS[0]:-}" && -n "${RANK_HOSTS[1]:-}" ]] || die "NODES must list exactly two targets, got: $NODES"
}

fabric_pinned() {
  [[ -n "$RANK0_IP" && -n "$RANK1_IP" && -n "$RANK0_HCA" && -n "$RANK1_HCA" &&
     -n "$NCCL_IB_GID_INDEX" && -n "$FABRIC_IF_0" && -n "$FABRIC_IF_1" ]]
}

fabric_partial_pin() {
  local set=0 v
  for v in "$RANK0_IP" "$RANK1_IP" "$RANK0_HCA" "$RANK1_HCA" "$NCCL_IB_GID_INDEX" "$FABRIC_IF_0" "$FABRIC_IF_1"; do
    [[ -n $v ]] && set=$((set + 1))
  done
  (( set > 0 )) && ! fabric_pinned
}

TOOLS_INSTALLED=0
install_tools() {  # once per invocation; both ranks in parallel
  need_hosts
  (( TOOLS_INSTALLED == 1 )) && return 0
  local r pids=() rc=0 p
  for r in 0 1; do
    (
      remote "$r" "mkdir -p '$TOOLS' '$BASE_DIR/run'; rm -f '$BASE_DIR/overlay.json'"
      rsync -a -e "ssh ${SSH_OPTS[*]}" "$ROOT/scripts/" "$(target_of "$r"):$TOOLS/"
      if [[ -n "$OVERLAY_FILE" ]]; then rsync_to "$r" "$ROOT/$OVERLAY_FILE" "$BASE_DIR/overlay.json"; fi
      rsync_to "$r" "$ROOT/$MANIFEST_FILE" "$BASE_DIR/pack-manifest.tsv"
      remote "$r" "chmod +x '$TOOLS'/*.sh '$TOOLS'/*.py 2>/dev/null || true"
    ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || die "installing tools on the nodes failed"
  TOOLS_INSTALLED=1
}

write_pinned_plan() {
  mkdir -p "$RUN_DIR"
  python3 - "$RUN_DIR/fabric.json" "$RANK0_IP" "$RANK1_IP" "$RANK0_HCA" "$RANK1_HCA" \
    "$NCCL_IB_GID_INDEX" "$FABRIC_IF_0" "$FABRIC_IF_1" <<'PY'
import json, sys
out, ip0, ip1, hca0, hca1, gid, if0, if1 = sys.argv[1:9]
json.dump({
    "subnets": [], "gid": int(gid), "rendezvous": ip0, "mtu_uniform": True, "warnings": ["pinned"],
    "rank": [
        {"node": "rank0", "ip": ip0, "ifname": if0, "hcas": hca0.split(","), "mtu": 9000},
        {"node": "rank1", "ip": ip1, "ifname": if1, "hcas": hca1.split(","), "mtu": 9000},
    ],
}, open(out, "w"), indent=1, sort_keys=True)
PY
}

fabric_plan_now() {
  need_hosts
  mkdir -p "$RUN_DIR"
  if fabric_pinned; then
    write_pinned_plan
    return
  fi
  fabric_partial_pin && die "fabric pins are partial: set all seven (RANK0_IP, RANK1_IP, RANK0_HCA, RANK1_HCA, NCCL_IB_GID_INDEX, FABRIC_IF_0, FABRIC_IF_1) or none"
  install_tools
  remote 0 "python3 '$TOOLS/fabric_probe.py'" >"$RUN_DIR/fabric-rank0.json"
  remote 1 "python3 '$TOOLS/fabric_probe.py'" >"$RUN_DIR/fabric-rank1.json"
  python3 "$ROOT/scripts/fabric_plan.py" --rank0 "$RUN_DIR/fabric-rank0.json" \
    --rank1 "$RUN_DIR/fabric-rank1.json" --out "$RUN_DIR/fabric.json"
}

plan_file() {
  local f="${FABRIC_PLAN:-$RUN_DIR/fabric.json}"
  [[ -f "$f" ]] || die "no fabric plan at $f: run ./m31.sh doctor (or pin the seven FABRIC keys)"
  echo "$f"
}

# ── derived serving values ──────────────────────────────────────────────────
derive_config() {
  [[ -n "$MAX_MODEL_LEN" ]] || die "MAX_MODEL_LEN is empty (profile $PROFILE did not set it)"
  [[ -n "$MAX_NUM_SEQS" ]] || die "MAX_NUM_SEQS is empty (profile $PROFILE did not set it)"

  if [[ -z "$KVB" ]]; then
    KVB=$(python3 - "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$KV_BYTES_PER_TOKEN" <<'PY'
import sys
maxlen, seqs, per = map(int, sys.argv[1:4])
want = maxlen * seqs * per * 102 // 100
align = 256 * 1024 * 1024
print((want + align - 1) // align * align)
PY
)
  fi
  if [[ -z "$CAPTURE_SIZES" ]]; then
    CAPTURE_SIZES=$(python3 - "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$DSPARK" <<'PY'
import sys
seqs, k, dspark = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
rows = seqs * (k + 1) if dspark else seqs
base = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96]
sizes = [s for s in base if s <= rows]
if not sizes or sizes[-1] != rows:
    sizes.append(rows)
print(",".join(str(s) for s in sorted(set(sizes))))
PY
)
  fi
  MAX_CAPTURE="${CAPTURE_SIZES##*,}"

  (( MAX_NUM_BATCHED_TOKENS % 128 == 0 )) || die "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS must be a multiple of block size 128"
  [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || die "MAX_MODEL_LEN must be an integer"
}

check_config() {
  derive_config
  if [[ "$KV_Q4" == 1 ]]; then
    [[ "$KV_CACHE_DTYPE" == nvfp4 ]] || die "KV_Q4=1 requires KV_CACHE_DTYPE=nvfp4"
    [[ "$INDEXER_KV_DTYPE" == fp8 ]] || die "KV_Q4=1 requires INDEXER_KV_DTYPE=fp8"
    [[ "$DSPARK" == 1 && "$DRAFT_KV_DTYPE" == fp8 && "$DRAFT_ATTN_BACKEND" == TRITON_ATTN ]] \
      || die "KV_Q4=1 requires DSPARK=1, DRAFT_KV_DTYPE=fp8, DRAFT_ATTN_BACKEND=TRITON_ATTN"
  else
    [[ "$KV_CACHE_DTYPE" != nvfp4 ]] || die "KV_CACHE_DTYPE=nvfp4 without KV_Q4=1 is an upstream raise"
  fi
  if [[ "$DSPARK" == 1 ]]; then
    (( NUM_SPECULATIVE_TOKENS >= 1 )) || die "NUM_SPECULATIVE_TOKENS must be >= 1"
  fi
  if ! is_true "$PROFILE_MEASURED" && [[ -z "${M31_PROFILE_WARNED:-}" ]]; then
    log "WARNING: profile $PROFILE is UNMEASURED; do not publish its numbers"
    export M31_PROFILE_WARNED=1   # child print-serve calls inherit it and stay quiet
  fi
  [[ -z "$OVERLAY_FILE" || -f "$ROOT/$OVERLAY_FILE" ]] || die "overlay missing: $OVERLAY_FILE"
  [[ -f "$ROOT/$MANIFEST_FILE" ]] || die "manifest missing: $MANIFEST_FILE"
  if [[ "$MANIFEST_FILE" == "$LANE_MANIFEST_FILE" ]]; then
    local manifest_bytes
    manifest_bytes=$(awk -F'\t' '{s+=$2} END{printf "%d", s}' "$ROOT/$MANIFEST_FILE")
    [[ "$manifest_bytes" == "$MODEL_BYTES" ]] \
      || die "manifest totals $manifest_bytes bytes, pin says $MODEL_BYTES"
  fi
  local p
  for p in "${PATCH_LIST[@]}" ${EXL3_PATCH_LIST[@]+"${EXL3_PATCH_LIST[@]}"}; do [[ -f "$ROOT/$p" ]] || die "patch missing: $p"; done
  [[ "$EXL3_REV" =~ ^[0-9a-f]{40}$ && "$PLUGIN_REV" =~ ^[0-9a-f]{40}$ ]] \
    || die "EXL3_REV and PLUGIN_REV must be full 40-character commit shas (the build fetches exactly that commit)"
}

# ── commands ────────────────────────────────────────────────────────────────
cmd_print_serve() {
  local rank=${1:?rank}
  [[ "$rank" == 0 || "$rank" == 1 ]] || die "rank must be 0 or 1"
  fabric_partial_pin && die "fabric pins are partial: set all seven FABRIC keys or none"
  check_config
  local plan; plan=$(plan_file)
  local ip hcas ifname gid rv
  ip=$(jsonq "$plan" "rank.$rank.ip")
  hcas=$(jsonq "$plan" "rank.$rank.hcas")
  ifname=$(jsonq "$plan" "rank.$rank.ifname")
  gid=$(jsonq "$plan" gid)
  rv=$(jsonq "$plan" rendezvous)

  local guard_line
  guard_line="sudo -n setsid nohup bash $(printf '%q' "$TOOLS/memguard.sh") watch $(printf '%q' "$CONTAINER") $(printf '%q' "$BASE_DIR/run/guard.log") $(printf '%q' "$BASE_DIR/run/stage") $((GUARD_START_MIB * 1024)) $((GUARD_MIB * 1024)) $((GUARD_SWAP_MAX_MIB * 1024)) $DROP_CACHES_EVERY_S $DROP_CACHES_LOADING >/dev/null 2>&1 </dev/null &"

  env=(
    -e "VLLM_EXL3_MODEL_DIR=/models/$MODEL_NAME"
    -e "VLLM_HOST_IP=$ip"
    -e "NCCL_IB_HCA=$hcas"
    -e "NCCL_IB_GID_INDEX=$gid"
    -e "NCCL_SOCKET_IFNAME=$ifname"
    -e "GLOO_SOCKET_IFNAME=$ifname"
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e "VLLM_EXL3_Q4KV=$KV_Q4"
    -e "VLLM_EXL3_INDEXER_MXFP8=$INDEXER_MXFP8"
    -e MIMALLOC_PURGE_DELAY=0 -e MIMALLOC_RESET_DELAY=0 -e MALLOC_ARENA_MAX=2
    -e VLLM_EXL3_TRIM_AFTER_LAYER=1 -e VLLM_EXL3_UMA_POSTLOAD=1
    -e VLLM_EXL3_ATS_ALIAS=1 -e VLLM_EXL3_ATS_ALIAS_EMBED=1
    -e VLLM_EXL3_DRAFT_VOCAB_META=1 -e VLLM_EXL3_TRIM_BEFORE_KV=1
    -e "VLLM_EXL3_MOE_TEMP_ROWS=$MOE_TEMP_ROWS" -e VLLM_EXL3_PERIODIC_TRIM_S=30
    -e "VLLM_EXL3_FAT_THRESHOLD=$FAT_THRESHOLD" -e "VLLM_EXL3_FAT_THRESHOLD_PREFILL=$FAT_THRESHOLD_PREFILL" -e "VLLM_EXL3_FAT_PREFILL_TOKENS=$FAT_PREFILL_TOKENS"
    -e VLLM_EXL3_MADV_AFTER_H2D=0
    # Wide (128-column, 4-way k-split) coop tile on both MoE decode stages: 8-15% faster coop
    # launches on this pack's per-rank shapes (exllamav3 picks narrow for K=1536 by default).
    -e EXL3_MOE_COOP_WIDE=1
    -e "VLLM_EXL3_WEIGHT_SNAPSHOT=$WEIGHT_SNAPSHOT" -e VLLM_EXL3_WEIGHT_SNAPSHOT_DIR=/cache/weight-snapshots
    -e VLLM_TRITON_JIT_WARMUP_NUM_THREADS=1 -e TORCHINDUCTOR_COMPILE_THREADS=1
    -e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/nv-compute
    -e CUDA_CACHE_MAXSIZE=4294967296 -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm
  )
  [[ -n "$NCCL_DEBUG" ]] && env+=(-e "NCCL_DEBUG=$NCCL_DEBUG")
  [[ -n "$API_KEY" && "$rank" == 0 ]] && env+=(-e "VLLM_API_KEY=$API_KEY")
  local kv
  for kv in ${EXTRA_ENV:-}; do env+=(-e "$kv"); done

  serve=(serve "/models/$MODEL_NAME"
    --tensor-parallel-size 2 --nnodes 2 --node-rank "$rank"
    --master-addr "$rv" --master-port "$MASTER_PORT"
    --language-model-only --block-size 128 --disable-hybrid-kv-cache-manager
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-memory-bytes "$KVB" --kv-cache-dtype "$KV_CACHE_DTYPE"
    --attention-config "{\"indexer_kv_dtype\":\"$INDEXER_KV_DTYPE\"}"
    # MXFP8 linears (DSpark drafter, indexer) on Marlin: weight-only, bf16 activations, and at
    # decode M the drafter GEMMs run ~2x faster than flashinfer's cutlass MXFP8 (W8A8) kernel.
    --kernel-config '{"enable_flashinfer_autotune":false,"linear_backend_per_quant":{"mxfp8":"marlin"}}'
    --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice
    --override-generation-config "{\"max_new_tokens\": $DEFAULT_MAX_TOKENS}"
  )
  [[ "$rank" == 0 ]] && serve+=(--served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT")
  [[ "$rank" == 1 ]] && serve+=(--headless)
  if [[ "$DSPARK" == 1 ]]; then
    serve+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"attention_backend\":\"$DRAFT_ATTN_BACKEND\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"$DRAFT_SAMPLE\",\"rejection_sample_method\":\"$REJECTION_SAMPLE\"}")
  fi
  if is_true "$EAGER"; then
    serve+=(--enforce-eager)
  else
    serve+=(--compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$CAPTURE_SIZES],\"mode\":$COMPILE_MODE}"
            --max-cudagraph-capture-size "$MAX_CAPTURE")
  fi
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_VLLM_ARGS" ]] && serve+=($EXTRA_VLLM_ARGS)

  docker=(docker run -d --name "$CONTAINER"
    --network host --ipc host --gpus all
    --device /dev/infiniband --cap-add IPC_LOCK
    --ulimit memlock=-1:-1 --ulimit core=0 --shm-size 2g
    --memory="$CONTAINER_MEMORY" --memory-swap="$CONTAINER_MEMORY" --oom-score-adj 1000
    --security-opt seccomp=unconfined
  )
  [[ -n "$CPUSET" ]] && docker+=(--cpuset-cpus "$CPUSET")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_DOCKER_ARGS" ]] && docker+=($EXTRA_DOCKER_ARGS)
  docker+=(
    -v "$MODELS_DIR":/models:ro
    -v "$CACHE_DIR":/cache:rw
    -v "$BASE_DIR/serve-config.json":/models/"$MODEL_NAME"/config.json:ro
    "${env[@]}"
    --entrypoint vllm "$IMAGE"
    "${serve[@]}"
  )

  printf '%s\n' "$guard_line"
  printf '%s\n' "$(printf '%s\n' "${docker[@]}" | quoted)"
}

cmd_doctor() {
  need_hosts
  fabric_plan_now
  local plan; plan=$(plan_file)
  local fails=0 warns=0
  ok() { printf 'PASS %s\n' "$1"; }
  warn() { printf 'WARN %s\n' "$1"; warns=$((warns + 1)); }
  bad() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }

  local r node
  for r in 0 1; do
    node=$(host_of "$r")
    remote "$r" "true" || { bad "rank $r ($node) unreachable"; continue; }
    ok "rank $r ($node) ssh"

    local out
    out=$(remote "$r" '
      echo "host=$(hostname)"
      echo "mem_kib=$(awk "/^MemTotal:/{print \$2}" /proc/meminfo)"
      echo "avail_kib=$(awk "/^MemAvailable:/{print \$2}" /proc/meminfo)"
      echo "pagesize=$(getconf PAGESIZE)"
      echo "disk_kib=$(df -Pk / | awk "NR==2{print \$4}")"
      echo "docker=$(docker --version 2>/dev/null | head -1)"
      echo "rsync=$(rsync --version 2>/dev/null | head -1 | awk "{print \$3}")"
      echo "hf=$([ -x "${HF_BIN:-$HOME/venvs/hf/bin/hf}" ] && echo yes || echo no)"
      echo "sudo=$(sudo -n true 2>/dev/null && echo yes || echo no)"
      foreign=$(docker ps --format "{{.Names}}" 2>/dev/null | tr "\n" " " || true)
      echo "foreign=$foreign"
      img=$(docker image inspect "$IMAGE" --format "{{.Id}}" 2>/dev/null) || img=missing
      echo "image=$img"
      last=$(docker logs --tail 1 "$CONTAINER" 2>/dev/null | tr "\n" " " | cut -c1-120 || true)
      echo "sys_last=$last"
      echo "compaction=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null)"
      echo "watermark=$(cat /proc/sys/vm/watermark_boost_factor 2>/dev/null)"
    ' IMAGE="$IMAGE" CONTAINER="$CONTAINER" HF_BIN="$HF_BIN")
    local host mem avail pagesize disk docker_ver rsync_ver hf sudo_ok foreign image compaction watermark
    host=$(sed -n 's/^host=//p' <<<"$out")
    mem=$(sed -n 's/^mem_kib=//p' <<<"$out")
    avail=$(sed -n 's/^avail_kib=//p' <<<"$out")
    pagesize=$(sed -n 's/^pagesize=//p' <<<"$out")
    disk=$(sed -n 's/^disk_kib=//p' <<<"$out")
    docker_ver=$(sed -n 's/^docker=//p' <<<"$out")
    rsync_ver=$(sed -n 's/^rsync=//p' <<<"$out")
    hf=$(sed -n 's/^hf=//p' <<<"$out")
    sudo_ok=$(sed -n 's/^sudo=//p' <<<"$out")
    foreign=$(sed -n 's/^foreign=//p' <<<"$out")
    image=$(sed -n 's/^image=//p' <<<"$out")
    compaction=$(sed -n 's/^compaction=//p' <<<"$out")
    watermark=$(sed -n 's/^watermark=//p' <<<"$out")

    printf '  %s: MemTotal %.1f GiB, MemAvailable %.1f GiB, disk free %.0f GiB, page size %s\n' \
      "$host" "$(awk -v k="$mem" 'BEGIN{print k/1048576}')" "$(awk -v k="$avail" 'BEGIN{print k/1048576}')" \
      "$(awk -v k="$disk" 'BEGIN{print k/1048576}')" "$pagesize"
    (( mem > 118 * 1024 * 1024 )) && ok "rank $r memory total" || bad "rank $r MemTotal ${mem} KiB (< 118 GiB)"
    # Pack 111 GiB + first-boot weight snapshot 56 GiB + image 27 GiB + caches (~196 GiB measured),
    # plus room for build layers and one older image during an upgrade.
    (( disk > 230 * 1024 * 1024 )) && ok "rank $r disk >= 230 GiB free" || bad "rank $r disk free ${disk} KiB (< 230 GiB)"
    [[ "$pagesize" == 4096 ]] && ok "rank $r 4 KiB pages" || warn "rank $r page size $pagesize (the recipe is qualified on 4 KiB)"
    [[ -n "$docker_ver" ]] && ok "rank $r docker: $docker_ver" || bad "rank $r docker missing"
    [[ -n "$rsync_ver" ]] && ok "rank $r rsync" || bad "rank $r rsync missing"
    [[ "$hf" == yes ]] && ok "rank $r hf CLI" || bad "rank $r hf CLI missing (expected ${HF_BIN:-$HOME/venvs/hf/bin/hf})"
    [[ "$sudo_ok" == yes ]] && ok "rank $r passwordless sudo" || bad "rank $r needs passwordless sudo (memory guard)"
    local name allowed=0
    for name in $foreign; do
      [[ "$name" == "$CONTAINER" ]] && continue
      case " $ALLOWED_CONTAINERS " in *" $name "*) ;; *) allowed=1 ;; esac
    done
    (( allowed == 0 )) && ok "rank $r no foreign containers" || bad "rank $r runs containers outside ALLOWED_CONTAINERS: $foreign"
    [[ "$image" == missing ]] && warn "rank $r image $IMAGE not built yet" || ok "rank $r image present"
    # start sets both to 0 for the serve and stop restores the host's values.
    ok "rank $r vm.compaction_proactiveness=$compaction (start sets 0 while serving)"
    ok "rank $r vm.watermark_boost_factor=$watermark (start sets 0 while serving)"
  done

  # Fabric: shared subnets, RoCE state, ping across the cable, MTU.
  local gid rv mtu_uniform subnets
  gid=$(jsonq "$plan" gid); rv=$(jsonq "$plan" rendezvous)
  mtu_uniform=$(jsonq "$plan" mtu_uniform)
  subnets=$(jsonq "$plan" subnets)
  printf 'fabric: gid %s, rendezvous %s, subnets %s\n' "$gid" "$rv" "${subnets:-pinned}"
  local fr
  for fr in 0 1; do
    printf '  rank %s: cable ip %s, interface %s, HCAs %s\n' "$fr" "$(jsonq "$plan" "rank.$fr.ip")" \
      "$(jsonq "$plan" "rank.$fr.ifname")" "$(jsonq "$plan" "rank.$fr.hcas")"
  done
  [[ "$mtu_uniform" == True ]] && ok "fabric MTU uniform" || warn "fabric MTU differs between ends"
  local ip0 ip1
  ip0=$(jsonq "$plan" rank.0.ip); ip1=$(jsonq "$plan" rank.1.ip)
  if remote 0 "ping -c1 -W2 '$ip1'" >/dev/null 2>&1; then ok "rank 0 pings rank 1 across the cable ($ip1)"; else bad "rank 0 cannot ping rank 1 at $ip1"; fi
  if remote 1 "ping -c1 -W2 '$ip0'" >/dev/null 2>&1; then ok "rank 1 pings rank 0 across the cable ($ip0)"; else bad "rank 1 cannot ping rank 0 at $ip0"; fi
  debug=$(remote 0 "for d in /sys/class/infiniband/*; do s=\$(cat \$d/ports/1/state); echo \"\$(basename \$d)=\$s\"; done" 2>/dev/null || true)
  if grep -q 'ACTIVE' <<<"$debug"; then ok "RoCE ports ACTIVE"; else bad "no ACTIVE RoCE port"; fi
  if remote 0 "ssh -n -o BatchMode=yes -o ConnectTimeout=5 '$SSH_USER@$ip1' true" >/dev/null 2>&1; then
    ok "rank 0 -> rank 1 cable ssh (weight/image fan-out)"
  else
    warn "rank 0 cannot ssh rank 1 at $ip1; fetch/build fall back to the management LAN (slower)"
  fi
  # build fetches both pinned engine commits from GitHub on rank 0.
  local repo
  for repo in "$EXL3_REPO" "$PLUGIN_REPO"; do
    if remote 0 "GIT_TERMINAL_PROMPT=0 timeout 20 git ls-remote --exit-code '$repo' HEAD" >/dev/null 2>&1; then
      ok "rank 0 reaches $repo (build source)"
    else
      bad "rank 0 cannot reach $repo; build fetches the pinned commit from there"
    fi
  done

  # Model pack: quick size check when present.
  for r in 0 1; do
    if remote "$r" "test -d '$MODEL_DIR'"; then
      if remote "$r" "python3 '$TOOLS/verify_manifest.py' --pack '$MODEL_DIR' --manifest '$BASE_DIR/pack-manifest.tsv' --quick" >/dev/null 2>&1; then
        ok "rank $r pack sizes match the manifest"
      else
        bad "rank $r pack is incomplete or has wrong file sizes (run ./m31.sh fetch)"
      fi
    else
      warn "rank $r has no pack at $MODEL_DIR (run ./m31.sh fetch)"
    fi
  done

  printf 'doctor: %d failure(s), %d warning(s)\n' "$fails" "$warns"
  (( fails == 0 )) || exit 1
}

cmd_fetch() {
  need_hosts
  check_config
  install_tools
  fabric_plan_now
  local plan; plan=$(plan_file)
  local cable_ip; cable_ip=$(jsonq "$plan" rank.1.ip)
  local peer=""
  if remote 0 "ssh -n -o BatchMode=yes -o ConnectTimeout=5 '$SSH_USER@$cable_ip' true" >/dev/null 2>&1; then
    peer="$SSH_USER@$cable_ip"
  else
    log "WARNING: cable ssh unavailable; staging over the management LAN"
    peer="$(target_of 1)"
  fi
  mkdir -p "$RUN_DIR"
  python3 "$ROOT/scripts/split_manifest.py" --manifest "$ROOT/$MANIFEST_FILE" --parts 2 --out-dir "$RUN_DIR/parts"
  local r
  for r in 0 1; do
    rsync_to "$r" "$RUN_DIR/parts/part0.txt" "$BASE_DIR/run/part0.txt"
    rsync_to "$r" "$RUN_DIR/parts/part1.txt" "$BASE_DIR/run/part1.txt"
  done

  local hf="${HF_BIN:-/home/$SSH_USER/venvs/hf/bin/hf}"
  log "downloading split parts in parallel (rank 0: part 0, rank 1: part 1)"
  local pids=()
  for r in 0 1; do
    (
      remote "$r" '
        set -euo pipefail
        if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi
        mkdir -p "$MODEL_DIR"
        mapfile -t inc < "$PART"
        [ "${#inc[@]}" -gt 0 ] || { echo "empty download part: $PART" >&2; exit 1; }
        args=()
        for p in "${inc[@]}"; do args+=(--include "$p"); done
        "$HF" download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR" "${args[@]}"
      ' MODEL_DIR="$MODEL_DIR" REPO="$MODEL_REPO" REV="$MODEL_REV" PART="$BASE_DIR/run/part$r.txt" HF="$hf" HF_TOKEN="$HF_TOKEN"
    ) >"$RUN_DIR/fetch-rank$r.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || { tail -20 "$RUN_DIR"/fetch-rank*.log >&2; die "download failed"; }

  log "gathering part 1 on rank 0 over the cable"
  remote 0 '
    set -euo pipefail
    rsync -a --files-from="$PART" -e "ssh -o BatchMode=yes" "$PEER:$MODEL_DIR/" "$MODEL_DIR/"
  ' MODEL_DIR="$MODEL_DIR" PEER="$peer" PART="$BASE_DIR/run/part1.txt"

  log "fanning part 0 out to rank 1"
  remote 0 '
    set -euo pipefail
    rsync -a --files-from="$PART" -e "ssh -o BatchMode=yes" "$MODEL_DIR/" "$PEER:$MODEL_DIR/"
  ' MODEL_DIR="$MODEL_DIR" PEER="$peer" PART="$BASE_DIR/run/part0.txt"

  log "verifying every file on both nodes (sha256)"
  both "
    python3 '$TOOLS/verify_manifest.py' --pack '$MODEL_DIR' --manifest '$BASE_DIR/pack-manifest.tsv' --json '$BASE_DIR/run/verify.json'
  " MODEL_DIR="$MODEL_DIR"
  log "fetch complete: $MODEL_DIR ($(awk -F'\t' '{s+=$2} END{printf "%.1f GiB", s/1073741824}' "$ROOT/$MANIFEST_FILE")) per node"
}

cmd_verify() {
  need_hosts
  install_tools
  both "
    python3 '$TOOLS/verify_manifest.py' --pack '$MODEL_DIR' --manifest '$BASE_DIR/pack-manifest.tsv'
  " MODEL_DIR="$MODEL_DIR"
}

cmd_build() {
  need_hosts
  check_config
  install_tools
  if remote 0 "docker inspect -f '{{.State.Running}}' '$CONTAINER' 2>/dev/null | grep -q true"; then
    is_true "${FORCE:-0}" || die "$CONTAINER is running on rank 0; ./m31.sh stop first (or FORCE=1)"
  fi
  local avail
  avail=$(remote 0 "awk '/MemAvailable/{print \$2}' /proc/meminfo")
  (( avail > 60 * 1024 * 1024 )) || die "rank 0 has $(awk -v k="$avail" 'BEGIN{printf "%.1f", k/1048576}') GiB available; the build needs ~60 GiB"

  remote 0 "mkdir -p '$BUILD_DIR/docker' '$BUILD_DIR/src'"
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$ROOT/docker/" "$(target_of 0):$BUILD_DIR/docker/"
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$ROOT/patches/" "$(target_of 0):$BUILD_DIR/patches/"
  rsync_to 0 "$ROOT/scripts/prepare_src.sh" "$BUILD_DIR/prepare_src.sh"

  log "pulling the pinned base image on rank 0"
  remote 0 "docker pull '$BASE_IMAGE@$BASE_IMAGE_DIGEST'" >/dev/null
  log "staging pinned trees and building $IMAGE on rank 0"
  remote 0 '
    set -euo pipefail
    export EXL3_REPO EXL3_REV PLUGIN_REPO PLUGIN_REV BASE_IMAGE_DIGEST
    EXL3_PATCHES=$(for p in $EXL3_PATCHES_REL; do printf "%s " "$BUILD_DIR/patches/$p"; done); export EXL3_PATCHES
    cd "$BUILD_DIR"
    bash prepare_src.sh "$BUILD_DIR/src" "$BUILD_DIR" $(for p in $PATCHES; do printf "%s " "$BUILD_DIR/patches/$p"; done)
    docker build -t "$IMAGE" --build-arg BASE_IMAGE_DIGEST="$BASE_IMAGE_DIGEST" \
      --build-arg EXL3_REV="$EXL3_REV" --build-arg PLUGIN_REV="$PLUGIN_REV" \
      --build-arg PATCH_SHA256="$PATCH_SHA256" \
      "$BUILD_DIR/docker"
  ' BUILD_DIR="$BUILD_DIR" EXL3_REPO="$EXL3_REPO" EXL3_REV="$EXL3_REV" \
    PLUGIN_REPO="$PLUGIN_REPO" PLUGIN_REV="$PLUGIN_REV" \
    BASE_IMAGE_DIGEST="$BASE_IMAGE_DIGEST" IMAGE="$IMAGE" PATCH_SHA256="$PATCH_SHA256" \
    PATCHES="$(for p in "${PATCH_LIST[@]}"; do printf '%s ' "$(basename "$p")"; done)" \
    EXL3_PATCHES_REL="$(for p in ${EXL3_PATCH_LIST[@]+"${EXL3_PATCH_LIST[@]}"}; do printf '%s ' "${p#patches/}"; done)"

  local cable_ip
  fabric_plan_now
  cable_ip=$(jsonq "$(plan_file)" rank.1.ip)
  if remote 0 "ssh -n -o BatchMode=yes -o ConnectTimeout=5 '$SSH_USER@$cable_ip' true" >/dev/null 2>&1; then
    log "shipping the image to rank 1 over the cable"
    remote 0 "docker save '$IMAGE' | ssh -o BatchMode=yes '$SSH_USER@$cable_ip' 'docker load'"
  else
    log "WARNING: cable ssh unavailable; shipping the image over the management LAN"
    remote 0 "docker save '$IMAGE' | ssh -o BatchMode=yes '$(target_of 1)' 'docker load'"
  fi
  both "docker image inspect '$IMAGE' --format '{{.Id}}'" IMAGE="$IMAGE"
  log "built $IMAGE"
}

now_s() { printf '%s' "${EPOCHREALTIME:-$(date +%s)}"; }
since() { awk -v a="$1" -v b="$(now_s)" 'BEGIN{printf "%.1f", b - a}'; }

cmd_start() {
  local t_start; t_start=$(now_s)
  need_hosts
  check_config
  install_tools
  fabric_plan_now
  local plan; plan=$(plan_file)
  local overlay_sha=""
  [[ -z "$OVERLAY_FILE" ]] || overlay_sha=$(sha256_file "$ROOT/$OVERLAY_FILE")
  # One ssh per rank, both in parallel: every value the checks below need.
  local r
  for r in 0 1; do
    remote "$r" '
      if test -d "$MODEL_DIR"; then echo pack_dir=1; else echo pack_dir=0; fi
      if python3 "$TOOLS/verify_manifest.py" --pack "$MODEL_DIR" --manifest "$BASE_DIR/pack-manifest.tsv" --quick >/dev/null 2>&1; then echo pack_sizes=1; else echo pack_sizes=0; fi
      if [ -f "$BASE_DIR/overlay.json" ]; then echo "overlay_sha=$(sha256sum "$BASE_DIR/overlay.json" | cut -d" " -f1)"; else echo overlay_sha=; fi
      # shellcheck disable=SC2086
      if python3 "$TOOLS/make_serve_config.py" --base "$MODEL_DIR/config.json" $OVERLAY_ARG --out "$BASE_DIR/serve-config.json" >/dev/null 2>&1; then
        echo "serve_sha=$(sha256sum "$BASE_DIR/serve-config.json" | cut -d" " -f1)"
      else
        echo serve_sha=FAIL
      fi
      echo "labels=$(docker image inspect "$IMAGE" --format "{{index .Config.Labels \"local.m31.base-digest\"}} {{index .Config.Labels \"local.m31.exl3\"}} {{index .Config.Labels \"local.m31.plugin\"}} {{index .Config.Labels \"local.m31.patch-sha256\"}}" 2>/dev/null || echo missing)"
      echo "foreign=$(docker ps --format "{{.Names}}" | tr "\n" " ")"
    ' MODEL_DIR="$MODEL_DIR" TOOLS="$TOOLS" BASE_DIR="$BASE_DIR" IMAGE="$IMAGE" \
      OVERLAY_ARG="${overlay_sha:+--overlay $BASE_DIR/overlay.json}" >"$RUN_DIR/preflight-rank$r.txt" 2>&1 &
  done
  wait
  pf() { sed -n "s/^$2=//p" "$RUN_DIR/preflight-rank$1.txt" | tail -1; }
  local serve0 name allowed foreign
  serve0=$(pf 0 serve_sha)
  for r in 0 1; do
    [[ "$(pf "$r" pack_dir)" == 1 ]] || die "rank $r has no pack at $MODEL_DIR (./m31.sh fetch)"
    [[ "$(pf "$r" pack_sizes)" == 1 ]] || die "rank $r pack sizes do not match the manifest"
    if [[ -n "$overlay_sha" ]]; then
      [[ "$(pf "$r" overlay_sha)" == "$overlay_sha" ]] || die "rank $r overlay differs from $OVERLAY_FILE"
    fi
    [[ "$(pf "$r" serve_sha)" != FAIL && -n "$(pf "$r" serve_sha)" ]] || die "rank $r could not build the serve config"
    [[ "$(pf "$r" serve_sha)" == "$serve0" ]] || die "rank 1 serve config differs from rank 0"
    [[ "$(pf "$r" labels)" != missing ]] || die "rank $r lacks image $IMAGE (./m31.sh build)"
    [[ "$(pf "$r" labels)" == "$BASE_IMAGE_DIGEST $EXL3_REV $PLUGIN_REV $PATCH_SHA256" ]] \
      || die "rank $r image labels '$(pf "$r" labels)' do not match the pins (base $BASE_IMAGE_DIGEST, exl3 $EXL3_REV, plugin $PLUGIN_REV, patch sha $PATCH_SHA256)"
    # Foreign containers (anything not in ALLOWED_CONTAINERS) must be stopped first.
    foreign=$(pf "$r" foreign); allowed=1
    for name in $foreign; do
      [[ "$name" == "$CONTAINER" ]] && continue
      case " $ALLOWED_CONTAINERS " in *" $name "*) ;; *) allowed=0 ;; esac
    done
    (( allowed == 1 )) || die "rank $r runs foreign containers: $foreign (ALLOWED_CONTAINERS='$ALLOWED_CONTAINERS')"
  done

  # Save host state, stop old ranks, drop caches.
  both '
    mkdir -p "$BASE_DIR/run" "$CACHE_DIR"
    : > "$BASE_DIR/run/sysctl.before"
    for k in vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness; do
      printf "%s=%s\n" "$k" "$(cat /proc/sys/${k//.//})" >> "$BASE_DIR/run/sysctl.before"
    done
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    sudo -n bash "$TOOLS/memguard.sh" stop "$BASE_DIR/run/guard.log" >/dev/null 2>&1 || true
    echo loading > "$BASE_DIR/run/stage"
    sudo -n rm -f "$BASE_DIR/run/guard.log.min"
    sudo -n sysctl -q vm.compaction_proactiveness=0 vm.watermark_boost_factor=0 vm.swappiness=0
    sudo -n swapoff -a 2>/dev/null || true
    sudo -n swapon -a 2>/dev/null || true
    if [ "$DROP_CACHES_START" = 1 ]; then sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; fi
    pkill -f "[p]refetch_pack.sh" 2>/dev/null || true
    : > "$BASE_DIR/run/prefetch.log"
    if [ "$PREFETCH" = 1 ]; then
      setsid nohup bash "$TOOLS/prefetch_pack.sh" "$MODEL_DIR" "$BASE_DIR/pack-manifest.tsv" \
        "$BASE_DIR/run/prefetch.log" "$PREFETCH_PARALLEL" >/dev/null 2>&1 </dev/null &
    fi
  ' BASE_DIR="$BASE_DIR" CACHE_DIR="$CACHE_DIR" CONTAINER="$CONTAINER" TOOLS="$TOOLS" \
    MODEL_DIR="$MODEL_DIR" PREFETCH="$PREFETCH" PREFETCH_PARALLEL="$PREFETCH_PARALLEL" DROP_CACHES_START="$DROP_CACHES_START"

  # Both ranks' commands are built before either launches.
  local cmd0 cmd1; cmd1=$(cmd_print_serve 1); cmd0=$(cmd_print_serve 0)
  local t_launch; t_launch=$(now_s)
  if (( LAUNCH_STAGGER_S == 0 )); then
    log "launching both ranks in parallel"
    local p0 p1
    remote 1 "$cmd1" & p1=$!
    remote 0 "$cmd0" & p0=$!
    wait "$p1" || die "rank 1 launch failed"
    wait "$p0" || die "rank 0 launch failed"
  else
    log "launching rank 1 (headless), then rank 0 after ${LAUNCH_STAGGER_S}s"
    remote 1 "$cmd1" || die "rank 1 launch failed"
    sleep "$LAUNCH_STAGGER_S"
    remote 0 "$cmd0" || die "rank 0 launch failed"
  fi

  # Health is polled every second; container liveness every 10 s.
  local code=000 up=0 last=$SECONDS deadline=$((SECONDS + WAIT_TIMEOUT)) r
  while (( SECONDS < deadline )); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$URL/v1/models" 2>/dev/null) || true
    [[ "$code" == 200 ]] && { up=1; break; }
    if (( SECONDS - last >= 10 )); then
      last=$SECONDS
      for r in 0 1; do
        if ! remote "$r" "docker inspect -f '{{.State.Running}}' '$CONTAINER' 2>/dev/null | grep -q true"; then
          log "rank $r container exited; last log lines:"
          remote "$r" "docker logs --tail 60 '$CONTAINER' 2>&1" || true
          remote "$r" "tail -5 '$BASE_DIR/run/guard.log'" || true
          save_rank_logs
          cmd_stop || true
          die "start failed (rank $r; full logs in $RUN_DIR/rank0.log, rank1.log)"
        fi
        # The memory guard is mandatory: a rank without a live guard stops the start.
        if ! remote "$r" "p=\$(cat '$BASE_DIR/run/guard.log.pid' 2>/dev/null) && kill -0 \"\$p\" 2>/dev/null || sudo -n kill -0 \"\$p\" 2>/dev/null"; then
          log "rank $r has no live memory guard; guard log:"
          remote "$r" "tail -5 '$BASE_DIR/run/guard.log'" || true
          save_rank_logs
          cmd_stop || true
          die "start failed (rank $r unguarded)"
        fi
      done
      log "waiting for /v1/models ($(since "$t_launch")s since launch; code=$code)"
    fi
    sleep 1
  done
  (( up == 1 )) || { save_rank_logs; die "/v1/models never answered within ${WAIT_TIMEOUT}s (last code $code)"; }
  local endpoint_up_s engine_up_s
  endpoint_up_s=$(since "$t_start"); engine_up_s=$(since "$t_launch")

  both "echo ready > '$BASE_DIR/run/stage'" BASE_DIR="$BASE_DIR"
  local mem0 mem1 kvline
  mem0=$(remote 0 "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo")
  mem1=$(remote 1 "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo")
  kvline=$(remote 0 "docker logs '$CONTAINER' 2>&1 | grep -iE 'kv cache size|kv-cache memory|kv cache memory|GPU KV cache' | tail -3" || true)
  {
    echo "ready=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "t_start=$t_start"
    echo "t_launch=$t_launch"
    echo "preflight_s=$(awk -v a="$t_start" -v b="$t_launch" 'BEGIN{printf "%.1f", b - a}')"
    echo "endpoint_up_s=$endpoint_up_s"
    echo "engine_up_s=$engine_up_s"
    echo "mem_available_gib_rank0=$mem0"
    echo "mem_available_gib_rank1=$mem1"
    echo "lane=$LANE"
    echo "profile=$PROFILE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "kvb=$KVB"
    echo "image=$IMAGE"
  } >"$RUN_DIR/start.txt"
  printf '%s\n' "$kvline" >"$RUN_DIR/kv.txt"
  log "endpoint up after ${endpoint_up_s}s (engine ${engine_up_s}s after launch); ready MemAvailable rank0 ${mem0} GiB, rank1 ${mem1} GiB"
  [[ -n "$kvline" ]] && log "kv: $kvline"

  # Warm-up and smoke are independent requests and both finish before READY:
  # run them at once. Each one's own wall time is recorded.
  local t_mark wpid=""; t_mark=$(now_s)
  if is_true "$WARMUP"; then
    log "warm-up (short chat + ~2.4K-token prompt, max_tokens 32) alongside smoke"
    ( python3 "$ROOT/scripts/warmup.py" --base-url "$URL" --model "$SERVED_NAME" --run "$RUN_DIR" \
        && echo "warmup_s=$(since "$t_mark")" >>"$RUN_DIR/start.txt" ) & wpid=$!
  fi
  cmd_smoke || die "smoke failed"
  echo "smoke_s=$(since "$t_mark")" >>"$RUN_DIR/start.txt"
  if [[ -n "$wpid" ]]; then wait "$wpid" || die "warm-up failed (see $RUN_DIR/warmup.jsonl)"; fi
  echo "ready_s=$(since "$t_start")" >>"$RUN_DIR/start.txt"
  save_rank_logs
  echo "READY: $SERVED_NAME at $URL in $(sed -n 's/^ready_s=//p' "$RUN_DIR/start.txt")s (run dir $RUN_DIR)"
  echo "next : ./m31.sh bench all"
}

# Both ranks' full container logs into the run dir (after a start, and before any failure stop
# removes the containers).
save_rank_logs() {
  local r
  for r in 0 1; do
    remote "$r" "docker logs -t '$CONTAINER' 2>&1" >"$RUN_DIR/rank$r.log" || true
  done
}

cmd_stop() {
  need_hosts
  local r
  for r in 0 1; do
    remote "$r" '
      mkdir -p "$BASE_DIR/run"
      if docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker logs --tail 2000 "$CONTAINER" > "$BASE_DIR/run/last.log" 2>&1 || true
      fi
      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      sudo -n bash "$TOOLS/memguard.sh" stop "$BASE_DIR/run/guard.log" >/dev/null 2>&1 || true
      if [ -s "$BASE_DIR/run/sysctl.before" ]; then
        while IFS="=" read -r k v; do
          case "$k" in vm.compaction_proactiveness|vm.watermark_boost_factor|vm.swappiness) sudo -n sysctl -q "$k=$v" >/dev/null 2>&1 || true ;; esac
        done < "$BASE_DIR/run/sysctl.before"
      fi
      sudo -n swapon -a 2>/dev/null || true
    ' BASE_DIR="$BASE_DIR" CONTAINER="$CONTAINER" TOOLS="$TOOLS" || log "rank $r stop reported an error"
  done
  log "stopped $CONTAINER on both nodes"
}

cmd_smoke() {
  python3 "$ROOT/tests/smoke.py" --base-url "$URL/v1" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} \
    || return 1
  local mode
  mode=$(remote 0 "docker logs '$CONTAINER' 2>&1 | grep -F 'EXL3 mixed-K MoE dispatch mode:' | head -1" || true)
  case "$mode" in
    *"dispatch mode: unified"*) echo "PASS mixed-K dispatch: unified" ;;
    *"dispatch mode: grouped"*) echo "FAIL mixed-K dispatch is grouped; this recipe never sets VLLM_EXL3_MIXEDK_MODE" >&2; return 1 ;;
    *) echo "FAIL no 'EXL3 mixed-K MoE dispatch mode:' line in rank 0's log" >&2; return 1 ;;
  esac
}

cmd_api_test() {
  python3 "$ROOT/tests/api_conformance.py" --base-url "$URL/v1" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} "$@"
}
cmd_probe() {
  python3 "$ROOT/bench/probe.py" --base-url "$URL" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} \
    --dspark "$DSPARK" "$@"
}
cmd_bench() {
  python3 "$ROOT/bench/bench.py" --base-url "$URL" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} \
    ${REFERENCE:+--reference "$ROOT/$REFERENCE"} "$@"
}
cmd_quality() {
  python3 "$ROOT/bench/quality_gate.py" "$URL" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} "$@"
}
cmd_longctx() {
  python3 "$ROOT/bench/longctx.py" --base-url "$URL" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} "$@"
}
cmd_tools() {
  python3 "$ROOT/bench/toolgate.py" --base-url "$URL/v1" --model "$SERVED_NAME" ${API_KEY:+--api-key "$API_KEY"} "$@"
}

cmd_status() {
  need_hosts
  both '
    echo "container: $(docker inspect -f "{{.Status}}" "$CONTAINER" 2>/dev/null || echo absent)"
    echo "MemAvailable: $(awk "/MemAvailable/{printf \"%.1f GiB\", \$2/1048576}" /proc/meminfo)"
    tail -1 "$BASE_DIR/run/guard.log" 2>/dev/null || true
  ' BASE_DIR="$BASE_DIR" CONTAINER="$CONTAINER"
  curl -s -m 5 "$URL/v1/models" || echo "endpoint: down"
  echo
}

cmd_logs() {
  need_hosts
  local rank=${1:-0} n=${2:-80}
  remote "$rank" "docker logs --tail '$n' '$CONTAINER' 2>&1" CONTAINER="$CONTAINER"
}

cmd_pin() { cmd_bench pin "$@"; }
cmd_check() { cmd_bench check "$@"; }

cmd_stop_other() { [[ -n "$STOP_OTHER" ]] || die "STOP_OTHER is not set (.env.example)"; bash -c "$STOP_OTHER"; }
cmd_restore_other() {
  [[ -n "$RESTORE_OTHER" ]] || die "RESTORE_OTHER is not set (.env.example)"
  cmd_stop || true
  bash -c "$RESTORE_OTHER"
}

cmd_help() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

# The whole dispatcher is one function and the script exits right after it:
# bash reads a script lazily, so editing this file while a long command runs
# must never make bash resume at a stale offset and run a stray fragment.
main() {
case ${1:-help} in
  doctor) cmd_doctor ;;
  fetch) cmd_fetch ;;
  verify) cmd_verify ;;
  build) cmd_build ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "$@" ;;
  smoke) cmd_smoke ;;
  probe) shift; cmd_probe "$@" ;;
  api-test) shift; cmd_api_test "$@" ;;
  bench) shift; cmd_bench "$@" ;;
  quality) shift; cmd_quality "$@" ;;
  longctx) shift; cmd_longctx "$@" ;;
  tools) shift; cmd_tools "$@" ;;
  pin) shift; cmd_pin "$@" ;;
  check) shift; cmd_check "$@" ;;
  print-serve) shift; cmd_print_serve "${1:-0}" ;;
  stop-other) cmd_stop_other ;;
  restore-other) cmd_restore_other ;;
  help|-h|--help) cmd_help ;;
  *) echo "unknown command: $1" >&2; cmd_help >&2; exit 2 ;;
esac
}
main "$@"; exit $?

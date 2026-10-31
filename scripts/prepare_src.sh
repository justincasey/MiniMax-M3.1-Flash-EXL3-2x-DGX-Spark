#!/usr/bin/env bash
# Stage the pinned engine trees and the build context on rank 0.
# Runs on the Spark, not the control machine; m31.sh ships it via ssh.
#
#   prepare_src.sh SRC_DIR BUILD_DIR PATCH_FILE...   (applied in order)
#
# Both engines are fetched from GitHub at their pinned full commit sha (shallow, cached in a bare
# repo under SRC_DIR so rebuilds don't refetch) and staged with `git archive`. Fail-closed: a
# commit that can't be fetched, or a patch that does not apply, aborts the build.
set -euo pipefail

SRC=${1:?src dir}
BUILD=${2:?build dir}
shift 2
(( $# >= 1 )) || { echo "prepare_src: at least one patch file" >&2; exit 1; }
PATCHES=("$@")
EXL3_REV=${EXL3_REV:?}
PLUGIN_REV=${PLUGIN_REV:?}
EXL3_REPO=${EXL3_REPO:?}
PLUGIN_REPO=${PLUGIN_REPO:?}

log() { printf 'prepare_src: %s\n' "$*" >&2; }
die() { log "FAIL $*"; exit 1; }

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'; }

# fetch_pinned NAME URL REV: fetch exactly REV (a full sha) into the bare cache $SRC/NAME.git,
# extract it to $SRC/NAME with `git archive`, and index the tree in a throwaway git repo so
# patches apply with `git apply` exactly as on a checkout of the pinned commit.
fetch_pinned() {
  local name=$1 url=$2 rev=$3 cache="$SRC/$1.git"
  [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || die "$name rev '$rev' is not a full 40-character commit sha"
  # --git-dir: a broken cache must not fall through to a parent repository. A full sha names exactly one
  # commit and tree, so fetching it by sha is the integrity check.
  [ -d "$cache" ] || git init -q --bare "$cache"
  if ! git --git-dir="$cache" cat-file -e "$rev^{commit}" 2>/dev/null; then
    GIT_TERMINAL_PROMPT=0 git --git-dir="$cache" fetch -q --depth 1 "$url" "$rev" \
      || die "cannot fetch $name $rev from $url (if the network is fine, remove $cache and retry)"
  fi
  rm -rf "${SRC:?}/$name"
  mkdir -p "$SRC/$name"
  git --git-dir="$cache" archive "$rev" | tar -x -C "$SRC/$name"
  git -C "$SRC/$name" init -q
  git -C "$SRC/$name" add -A
  log "fetched $name $rev from $url"
}

mkdir -p "$SRC" "$BUILD"

# ── exllamav3: pinned EXL3_REV + the exllamav3 patch series ──────────────────
fetch_pinned exllamav3 "$EXL3_REPO" "$EXL3_REV"
# exllamav3 patch series (EXL3_PATCHES, space-separated paths), applied to the index so the
# export below is tracked-only.
read -r -a EXL3_PATCH_ARR <<<"${EXL3_PATCHES:-}"
for PATCH in ${EXL3_PATCH_ARR[@]+"${EXL3_PATCH_ARR[@]}"}; do
  [ -f "$PATCH" ] || die "exllamav3 patch not found: $PATCH"
  git -C "$SRC/exllamav3" apply --index --check "$PATCH" 2>/dev/null || die "$(basename "$PATCH") does not apply to exllamav3 $EXL3_REV (+ earlier patches)"
  git -C "$SRC/exllamav3" apply --index "$PATCH"
  log "applied exllamav3 $(basename "$PATCH")"
done

# ── vllm-exl3: pinned PLUGIN_REV + the recipe patch series ────────────────────
fetch_pinned vllm-exl3 "$PLUGIN_REPO" "$PLUGIN_REV"

for PATCH in "${PATCHES[@]}"; do
  [ -f "$PATCH" ] || die "patch not found: $PATCH"
  git -C "$SRC/vllm-exl3" apply --check "$PATCH" 2>/dev/null || die "$(basename "$PATCH") does not apply to $PLUGIN_REV (+ earlier patches)"
  git -C "$SRC/vllm-exl3" apply "$PATCH"
  log "applied $(basename "$PATCH")"
done
for marker in _lm_head_full_vocab _indexer_rows_for_rank 'getattr(layer, "use_ep", None) is False' \
    'elif moe_act == MOE_ACT_SWIGLU_OAI:' 'def _narrow_lm_head' 'native_any = bool('; do
  grep -qF "$marker" "$SRC/vllm-exl3/src/vllm_exl3/exl3.py" || die "patched tree lacks $marker"
done
grep -q EXL3_FAT_GEMM_MAX_K "$SRC/vllm-exl3/csrc/bindings.cpp" || die "patched tree lacks the any-K fat GEMM"
grep -q exl3_fat_gemm_grouped "$SRC/vllm-exl3/csrc/bindings.cpp" || die "patched tree lacks the grouped fat GEMM"
grep -q "def _apply_wide_prefill" "$SRC/vllm-exl3/src/vllm_exl3/exl3.py" || die "patched tree lacks the bf16 dense prefill"
grep -q "def _mixedk_m_tile" "$SRC/vllm-exl3/src/vllm_exl3/exl3.py" || die "patched tree lacks the MoE prefill kernels"
grep -q "_COOP_SPLIT_MAX_TOKENS" "$SRC/vllm-exl3/src/vllm_exl3/exl3.py" || die "patched tree lacks the mixed-K coop decode"
test -f "$SRC/vllm-exl3/src/vllm_exl3/sparse_attn_tiled.py" || die "patched tree lacks the tiled sparse attention"
grep -q "FAT_THRESHOLD_PREFILL" "$SRC/vllm-exl3/src/vllm_exl3/exl3.py" || die "patched tree lacks the fat-tier prefill threshold"
if grep -q "exl3_moe_coop_mk" "$SRC/vllm-exl3/src/vllm_exl3/exl3.py"; then
  grep -rq "exl3_moe_coop_mk" "$SRC/exllamav3/exllamav3/exllamav3_ext/bindings.cpp" \
    || die "the plugin calls exl3_moe_coop_mk but exllamav3 lacks it (EXL3_PATCH_FILES)"
fi
grep -q install_weight_snapshot "$SRC/vllm-exl3/src/vllm_exl3/__init__.py" \
  && [ -f "$SRC/vllm-exl3/src/vllm_exl3/weight_snapshot.py" ] || die "patched tree lacks the weight snapshot module"

# ── export pristine (tracked-only) trees into the Docker context ─────────────
CTX="$BUILD/docker"
mkdir -p "$CTX"
[ -f "$CTX/Dockerfile" ] || die "docker/Dockerfile was not staged by m31.sh"
rm -rf "$CTX/exllamav3-src" "$CTX/vllm-exl3-src"
mkdir -p "$CTX/exllamav3-src" "$CTX/vllm-exl3-src"
git -C "$SRC/exllamav3" archive "$(git -C "$SRC/exllamav3" write-tree)" | tar -x -C "$CTX/exllamav3-src"
# The plugin tree is patched on disk; export the working tree without .git.
tar -C "$SRC/vllm-exl3" --exclude=.git -cf - . | tar -x -C "$CTX/vllm-exl3-src"

{
  echo "base_image=vllm/vllm-openai:minimax-m31-flash"
  echo "base_digest=${BASE_IMAGE_DIGEST:-unknown}"
  echo "exllamav3_rev=$EXL3_REV"
  echo "exllamav3_source=$EXL3_REPO"
  echo "vllm_exl3_rev=$PLUGIN_REV"
  echo "vllm_exl3_source=$PLUGIN_REPO"
  for PATCH in "${PATCHES[@]}"; do echo "patch=$(basename "$PATCH") $(sha256_of "$PATCH")"; done
  for PATCH in ${EXL3_PATCH_ARR[@]+"${EXL3_PATCH_ARR[@]}"}; do echo "exllamav3_patch=$(basename "$PATCH") $(sha256_of "$PATCH")"; done
  series=$(mktemp); cat "${PATCHES[@]}" ${EXL3_PATCH_ARR[@]+"${EXL3_PATCH_ARR[@]}"} > "$series"
  echo "patch_series_sha256=$(sha256_of "$series")"; rm -f "$series"
} > "$CTX/PINS"

log "staged $BUILD (exllamav3 $EXL3_REV + ${#EXL3_PATCH_ARR[@]} patches, plugin $PLUGIN_REV + ${#PATCHES[@]} patches)"

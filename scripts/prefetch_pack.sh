#!/usr/bin/env bash
# Read a pack's weight shards into the page cache, in load order, in the background.
# Runs on a Spark; `./m31.sh start` launches it before the containers when PREFETCH=1,
# so the reads overlap container start, imports and NCCL init.
#
#   prefetch_pack.sh MODEL_DIR MANIFEST LOG [PARALLEL]
#
# Order: the main shards by name (the loader walks keys in shard order), then the draft.
# Writes "prefetch: done <GiB> in <s>s" to LOG. Never fails a boot: any error is logged.
set -uo pipefail
dir=${1:?model dir} manifest=${2:?manifest} log=${3:?log} par=${4:-4}
t0=$(date +%s.%N)
mapfile -t files < <(cut -f3 "$manifest" | grep -E '^model-.*\.safetensors$' | sort; cut -f3 "$manifest" | grep -E '^[^/]+/.*\.safetensors$' | sort)
bytes=$(cut -f2,3 "$manifest" | awk -F'\t' '$2 ~ /\.safetensors$/ {s += $1} END {printf "%d", s}')
echo "$(date -Is) prefetch: ${#files[@]} files, $(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/2^30}') GiB, parallel $par" >>"$log"
printf '%s\n' "${files[@]}" | (cd "$dir" && xargs -P "$par" -I{} sh -c 'cat "$1" >/dev/null' _ {}) 2>>"$log"
echo "$(date -Is) prefetch: done in $(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')s" >>"$log"

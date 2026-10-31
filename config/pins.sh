# Release pins. Sourced by m31.sh and scripts/*.sh; never edited for tuning.
# Changing any value here is a release: re-verify the pack manifest, overlay,
# patch series and measured records together.
#
# Caller environment may override a pin for a one-off verification
# (PINS_FILE picks a different file); .env cannot, because this file is
# sourced before .env is parsed.

# Lane: which pack this release serves. Each lane file pins the pack (repo,
# revision, bytes, manifest, census, overlay) and its default profile.
# m31.sh resolves LANE from the caller environment or .env before this file.
LANE="${LANE:-2.15bpw}"
LANE_FILE="$ROOT/config/lanes/$LANE.sh"
[[ -f "$LANE_FILE" ]] || { echo "LANE=$LANE: no such lane at $LANE_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
. "$LANE_FILE"

# Base image: vllm/vllm-openai:minimax-m31-flash, pinned by the arm64 manifest
# list digest (the digest the one-Spark recipe measured on GB10).
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:minimax-m31-flash}"
BASE_IMAGE_DIGEST="${BASE_IMAGE_DIGEST:-sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c}"

# exllamav3: vcruz305's fork of turboderp-org/exllamav3 (MIT) at the act-3 (clamped
# SwiGLU-OAI) batched-MoE fix; a tree without it decodes garbage from every MiniMax MoE
# layer. The build fetches exactly this commit (full sha) and stages it with `git archive`.
EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3}"
EXL3_REV="${EXL3_REV:-fdd77361971144e8d175e09e7b50b7100739fd95}"
# exllamav3 patch series, applied in order on EXL3_REV. Part of the image tag's patch hash.
# 0001: coop MoE decode: unpredicated slice loop, early weight prefetch, device-side split-k.
# 0002: coop MoE decode: mixed-K instances, one launch per stage for a mixed-K layer.
# 0003: dense decode split-k GEMV (exl3_dgemv) for small weights at <= 16 rows (k/v, shared expert).
EXL3_PATCH_FILES="${EXL3_PATCH_FILES-patches/exllamav3/0001-exl3-coop-decode-prefetch-dsplit.patch patches/exllamav3/0002-exl3-coop-decode-mixedk-instances.patch patches/exllamav3/0003-exl3-dense-decode-split-k-gemv.patch}"

# vllm-exl3 plugin (vcruz305, AGPL-3.0-only): main at the q4 single-pass prefill merge.
# Fetched at PLUGIN_REV like exllamav3; the TP>1 enablement and every speed change are
# the patch series below, applied in order.
PLUGIN_REPO="${PLUGIN_REPO:-https://github.com/vcruz305/vllm-exl3}"
PLUGIN_REV="${PLUGIN_REV:-053c634db54afc981becfb46c0482e7358cc00ba}"

# 0001: the EXL3 LM head is 200,064 rows. At TP=2 that is 100,032 rows per
#       rank, which is not 128-aligned, so the plugin's padded-geometry guard
#       refuses it. The patch loads the whole 8-bit head per rank and returns
#       this rank's vocab columns. Derived from tonyd2wild's TP2 lane.
# 0002: the fused qkv+indexer under TP>1 when qkv is EXL3 (the 2.15 pack).
#       tonyd2wild's tp2-minimax-indexer.patch, byte-identical (sha256
#       8af3e860..., MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark @ 70c4e43).
# 0003: under TP without expert parallelism the plugin offset rank 1's expert
#       ids by ep.rank * num_experts, so its arena prescan missed every key and
#       each MoE layer fell back to stage-and-pack. Offset 0 when use_ep is off.
# 0004: per-rank weight snapshots (VLLM_EXL3_WEIGHT_SNAPSHOT=1): the first boot
#       of a configuration writes the loaded storages; later boots restore
#       them (O_DIRECT, CRC32 per chunk) instead of the ~90,000-tensor load.
# 0005: the fat-expert MoE tier (dequant + GEMM, used by large prefill chunks)
#       computed silu(gate) * up instead of the model's SwiGLU-OAI.
# 0006: under TP>1 each rank computes only the LM head blocks covering its
#       vocab partition instead of all 200,064 logits.
# 0007: the native fused fat GEMM accepts any trellis width (K 1..8) and
#       codebook, so mixed-K fat experts skip reconstruct + cuBLAS + index_add.
# 0008: grouped fat-expert prefill (one launch per K class per projection, fused
#       gather/activation Hadamards); opt-in FP8 dense prefill (VLLM_EXL3_FP8_PREFILL=1).
# 0009: bf16 dense prefill: one bf16 GEMM per run of EXL3 shards straight into the layer's
#       bf16 output (no fp16 copy, fp32 outputs, cat or cast); VLLM_EXL3_BF16_PREFILL=0 off.
# 0010: MoE prefill kernels: 64-row mixed-K tile for >= 2048-token batches; grouped fat GEMM
#       at two CTAs per SM with a vectorized epilogue.
# 0011: mixed-K MoE decode on exl3_moe_coop (one launch per K group, K-ordered pointer tables).
# 0012: query-tiled sparse attention for prefill-sized q4 calls.
# 0013: fat-tier prefill threshold (VLLM_EXL3_FAT_THRESHOLD_PREFILL, _FAT_PREFILL_TOKENS); tiled attention shape-agnostic JIT.
# 0014: tiled attention tile metadata on the GPU (no per-layer host sync).
# 0015: grouped fat GEMM on a cp.async multistage pipeline; empty M-blocks skipped.
# 0016: expert/wide dense GEMMs skipped in the profile run when kv_cache_memory_bytes is set.
# 0017: grouped fat GEMM full tiles without early exits (ldmatrix scheduled ahead of the MMAs).
# 0018: mixed-K MoE decode on one exl3_moe_coop_mk launch per stage (needs exllamav3 0002).
# 0019: prefill MoE glue: one token pass for both input Hadamards (+ zero fill), activation in the up epilogue.
# All are AGPL-3.0-only, like the plugin (patches/README.md).
PATCH_FILES="${PATCH_FILES:-patches/0001-minimax-exl3-lmhead-tp2.patch patches/0002-minimax-exl3-indexer-tp2.patch patches/0003-minimax-exl3-moe-prescan-tp-offset.patch patches/0004-minimax-exl3-weight-snapshots.patch patches/0005-minimax-exl3-fat-tier-swiglu-oai.patch patches/0006-minimax-exl3-lm-head-rank-blocks.patch patches/0007-minimax-exl3-fat-gemm-any-k.patch patches/0008-minimax-exl3-grouped-fat-fp8-prefill.patch patches/0009-minimax-exl3-bf16-dense-prefill.patch patches/0010-minimax-exl3-moe-prefill-kernels.patch patches/0011-minimax-exl3-mixedk-coop-decode.patch patches/0012-minimax-exl3-tiled-sparse-attention.patch patches/0013-minimax-exl3-fat-prefill-threshold.patch patches/0014-minimax-exl3-tiled-attention-no-sync.patch patches/0015-minimax-exl3-fat-grouped-cp-async.patch patches/0016-minimax-exl3-skip-profile-compute.patch patches/0017-minimax-exl3-fat-grouped-full-tile-loop.patch patches/0018-minimax-exl3-coop-mixedk-single-launch.patch patches/0019-minimax-exl3-fat-prefill-glue-fusion.patch}"

MANIFEST_FILE="${MANIFEST_FILE:-$LANE_MANIFEST_FILE}"

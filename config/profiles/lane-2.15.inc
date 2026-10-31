# Shared by the q4-215-* context profiles (sourced, not a profile itself). These knobs and
# pool sizes are measured for the 2.15 bpw lane only, so refuse any other lane.
[[ "${LANE:-}" == 2.15bpw ]] || die "PROFILE=$PROFILE is sized for LANE=2.15bpw (got LANE=${LANE:-unset})"

# Memory (2026-10-25, this recipe's image, 8 streams, 32 GiB KV pool per rank):
# - KV costs 27,520 bytes/token/rank (32 GiB -> 1,248,512 tokens); KV_BYTES_PER_TOKEN=28160
#   overestimates, so these profiles pin KVB instead of deriving it.
# - MemAvailable at ready: 12.2 GiB on the tightest test node (it also ran another container),
#   13.1-14.3 GiB elsewhere. A full-context prefill takes ~4.5 GiB more at 256K and 512K
#   alike: worst point 7.7 GiB (that node, 502K-token prompt), 8.7-10.0 GiB elsewhere.
# - Free memory moves 1:1 with the pool (preallocated). Pools are capped at 28 GiB, estimated
#   (32 GiB probe minus 4 GiB; full-context prefill not re-run at 28 GiB) to keep the worst
#   point at ~11.7 GiB on that node and holds 1,092,448 tokens: two full 512K
#   requests or four full 256K requests. 28 GiB = KVB 30064771072.

# Verified per profile (2026-10-25): boot 93-100 s; N concurrent decode streams with 0 errors;
# 64K needles 2/2; NLL check <= 6.88% (the dialogue passage's known 0 / 6.88% flip);
# guard minimum >= 12.1 GiB on rank 0 (14 GiB pool: 27.2 GiB); no guard kills.
# 8 streams decode slower in aggregate than 4 (verify batch 32 rows > the coop path's 24-token
# bound): 256K 55.8/72.8 vs 62.4/80.8 tok/s (prose/code), 512K 53.7 vs 62.3 prose.
# Full-context prompts are slow: attention cost grows with the prefix, so a 252K-token prompt
# took 14 min (~305 tok/s) and a 502K-token prompt 36 min (~230 tok/s) to first token.

# Prefill knobs tuned for this recipe (see q4-300k.env for the measurements).
: "${MAX_NUM_BATCHED_TOKENS:=16384}"
: "${FAT_THRESHOLD:=128}"
: "${FAT_THRESHOLD_PREFILL:=0}"
: "${FAT_PREFILL_TOKENS:=1024}"
# CAPTURE_SIZES is left to m31.sh's derivation: streams x (k+1) rows, k=3.

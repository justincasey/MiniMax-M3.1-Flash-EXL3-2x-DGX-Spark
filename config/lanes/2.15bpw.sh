# Lane 2.15bpw: vcruz305's SAGE 2.15 bpw pack at TP=2 (~55 GiB of weights per rank).
# Sourced by config/pins.sh when LANE=2.15bpw. Pins, not tunables.
#
# Every linear outside the routed experts is EXL3 (421 modules), and the fused
# sparse-attention indexer is MXFP8 inside the EXL3 QKV. The plugin's own
# config handling covers that, so there is no overlay: the serve config is the
# pack's config.json unchanged. patches/0002 lets the fused qkv+indexer shard.
#
# config/lanes/2.15bpw/pack-manifest.tsv pins the sha256 and size of every file
# at this revision; fetch verifies them.
MODEL_REPO="${MODEL_REPO:-vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw}"
MODEL_REV="${MODEL_REV:-99bf3a57af5d07245ea9abb5426e7dce6af6356b}"
MODEL_NAME="${MODEL_NAME:-MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw}"
MODEL_BYTES="${MODEL_BYTES:-118985525140}"
OVERLAY_FILE="${OVERLAY_FILE-}"
LANE_MANIFEST_FILE="config/lanes/2.15bpw/pack-manifest.tsv"
CENSUS_FILE="${CENSUS_FILE:-config/lanes/2.15bpw/pack-census.json}"
LANE_DEFAULT_PROFILE="q4-300k"

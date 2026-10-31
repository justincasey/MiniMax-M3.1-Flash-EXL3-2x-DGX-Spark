# Recipe decisions and traps

Decisions behind the 2.15 bpw TP=2 lane, each with its evidence, so the next recipe does not rediscover them.

## Model and pack

- **The pack's config is complete.** Every linear outside the routed experts is EXL3 (421 modules); the fused
  sparse-attention indexer is MXFP8 inside the EXL3 QKV. The plugin's own config handling covers that, so the serve
  config is the pack's `config.json` unchanged. `scripts/make_serve_config.py` still supports an overlay for packs
  that need one, and refuses to apply one twice.
- **Pin bytes, not only a revision.** vcruz305 once replaced the pack's history with a byte-identical tree. The
  sha256 per file in `config/lanes/2.15bpw/pack-manifest.tsv` is the authority; the Hub revision is a convenience.
- **The LFS sha256 is available from the Hub API** (`?blobs=true` → `siblings[].lfs.sha256`) without downloading.
- **A census needs no weights.** Safetensors headers sit at the start of each shard, so HTTP `Range` requests give
  dtype, shape and offset for every tensor in seconds (`scripts/pack_census.py`).

## TP=2

- **Two plugin refusals apply.**
  - The LM head: 200,064 rows split across two ranks is not 128-aligned. Patch 0001 loads the whole 8-bit head per
    rank, and 0006 computes only each rank's vocab blocks.
  - The EXL3 qkv with its fused indexer: patch 0002.
- **No expert offset without expert parallelism.** Under TP without EP the plugin offset rank 1's expert ids by
  `ep.rank * num_experts`, so its arena prescan missed every key (patch 0003).
- **Draft TP defaults to the target TP.** vLLM resolves an unset `draft_tensor_parallel_size` to the target TP, so
  the DSpark draft splits across both nodes with no extra flag.

## Memory

- **KV costs 27,520 bytes/token/rank** with q4 main KV, fp8 indexer and fp8 draft (a 32 GiB pool holds 1,248,512
  tokens). Pin `KVB` per profile rather than derive it.
- **A full-context prefill needs ~4.5 GiB beyond what is free at ready**, at 256K and 512K alike. Size the pool so
  that point stays well above the guard floor; see `config/profiles/lane-2.15.inc`.
- **The guard floor is not a context knob.** 2 GiB while loading, 1 GiB steady, kill after three consecutive
  breaches. Lower `KVB`, not the floor.
- **Fixed KV bytes make the profile run pure warm-up.** With `kv_cache_memory_bytes` set, vLLM still runs a full
  `max_num_batched_tokens` dummy forward. Patch 0016 skips its expert and wide dense GEMMs, saving about 6.5 s of
  boot at 16K chunks.

## Speed

- **Prefill chunks of 16K** are about 8% faster than 8K on 8-16K prompts. The cost: running decode streams stall
  for one chunk step (~11 s) while a long prompt prefills. 32K chunks were rejected: most of their gain on the
  benchmark prompt came from absorbing its 1.3K tail chunk, and the stall doubles.
- **In prefill, every routed expert takes the grouped fat GEMM** (`FAT_THRESHOLD_PREFILL=0`). In real routing, many
  experts get only a few rows; on the fat tier they fill otherwise idle SMs, at about a third of their mixed-K
  kernel cost.
- **Short batches keep `FAT_THRESHOLD=128`.** It also caps coop decode eligibility: a lower value pushed 65-128-token
  mixed batches off the coop kernel and cost about 10% at 4 streams.
- **Decode is bound by memory bandwidth.** The coop MoE kernels and the dense decode GEMMs run at 180-230 GB/s
  against a practical ~250 GB/s on GB10. 60 tok/s at 1 stream would need about half the bytes per token.
- **8 streams decode slower in total than 4.** The verify batch passes the coop path's 24-token bound.
- **Rejected, with numbers in the commit history of the patch series:**
  - NCCL `LL128` (slower), `Simple`, and channel caps (no effect);
  - dual-batch overlap (needs vLLM's V1 runner, which rejects DSpark);
  - pinning to the performance cores;
  - 256-row fat GEMM tiles;
  - FP8 dense prefill: it is opt-in and off, because its error is about 10× bf16's.

## Fabric

- **Map netdev → RDMA by GID IPv4, not sysfs.** On GB10 the P2p function's netdev has no `device/infiniband` link,
  but its RoCE v2 GID carries the same IPv4. Decoding the `::ffff:a.b.c.d` GIDs finds both functions.
- **A single cable exposes two /24s and two HCAs.** NCCL wants both HCAs (`rocep1s0f1,roceP2p1s0f1`) and one GID
  index.
- **Rendezvous over the cable, API over the management LAN.** `VLLM_HOST_IP` / `NCCL_SOCKET_IFNAME` use the cable;
  clients use the LAN address.
- **Node-to-node ssh over the cable** is used for the weight and image fan-out. `doctor` checks it, and the
  controller falls back to the management LAN with a warning.

## Operations

- **Remote helpers pass variables, not positional args.** `remote()` builds a prelude of `NAME=value`; a remote
  script that reads `$1` dies under `set -u`. A static grep in the contract test guards this.
- **The base image is a vLLM dev build** without an upstream commit label. `build` pulls it by digest and labels the
  result with the base, exllamav3, plugin and patch-series shas; `start` checks them.
- **The Docker build context is `$BUILD_DIR/docker`.** `prepare_src.sh` stages the two patched source trees and
  `PINS` there.
- **Save both ranks' logs before tearing down a failed start.** In TP the rank that exits is often a casualty; the
  cause is on its peer.

## Benchmarking

- **Tokens come from `usage`, not SSE chunks.** DSpark emits up to k+1 tokens per delta.
- **The MiniMax template is adaptive.** Bench with temperature 0 and no thinking flag; record reasoning tokens.
- **Spec acceptance is the health signal.** `bench spec` reports accepted draft tokens per verify step; each step
  emits one more token from the target. A value of 0 means the draft is rejected, not that the model is slow.
  Healthy at k=3: about 0.9 on prose and 1.7 on code.
- **Greedy output is not reproducible on this stack.** Near-ties in sparse-attention block selection flip early
  tokens. The quality gate is teacher-forced NLL with a 10% tolerance. Its short passages are hypersensitive: one
  passage flips between about 0% and 6.88% from run to run.

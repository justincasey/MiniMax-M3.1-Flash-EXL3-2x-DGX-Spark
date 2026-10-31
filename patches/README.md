# Patches

The plugin series (`patches/0001`-`0019`) is **AGPL-3.0-only**, because the plugin it patches is AGPL-3.0-only. The
exllamav3 series (`patches/exllamav3/`) is **MIT**, like exllamav3. The rest of the repository is Apache-2.0
(`LICENSE`); these files are not. `./m31.sh build` applies them in order (`PATCH_FILES` in
`config/pins.sh`), and the image tag and `local.m31.patch-sha256` label carry one sha256 over the series.

**How they are applied.** `./m31.sh build` fetches `vcruz305/vllm-exl3` at
`053c634db54afc981becfb46c0482e7358cc00ba` (pinned in `config/pins.sh`), exports it with `git archive`, then applies
`0001`-`0019` in order with `git apply --check` first. The exllamav3 series in `patches/exllamav3/` is applied the same
way to `vcruz305/exllamav3` at `fdd77361`. `python3 tests/test_pack_layout.py` runs the same staging step.

**Authors.** `0001` is derived from tonyd2wild's TP=2 LM-head change and `0002` is his file byte for byte
(MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark @ 70c4e43). The rest are this recipe's own. All apply to vcruz305's
`vllm-exl3`.

The base image `vllm/vllm-openai:minimax-m31-flash` is a vLLM dev build (`0.1.dev23341+g8744dd2fb`) whose commit is
not in public vLLM history, so the image digest is the only pin for the vLLM side.

## `0001-minimax-exl3-lmhead-tp2.patch`

**Target:** `vcruz305/vllm-exl3` at `053c634db54afc981becfb46c0482e7358cc00ba`
**License:** AGPL-3.0-only, because the plugin is AGPL-3.0-only.

**Problem.** EXL3 rotates a linear's output dimension in 128-row Hadamard blocks. MiniMax-M3.1's LM head has
200,064 rows, which is 128-aligned as a whole, but at TP=2 each rank's half is 100,032 rows, which is not. The
plugin refuses padded geometry under TP>1:

```
EXL3 padded linear geometry requires tensor parallel size 1
```

**Fix.** Under TP>1, every rank loads the *whole* 8-bit head (1.15 GiB, +0.57 GiB per node at TP=2) and `apply()`
returns this rank's vocab columns, padded to `num_embeddings_per_partition`. vLLM's logits gather is unchanged.
A marker on the layer (`_exl3_lm_full`) makes `_resolve_tp_geometry` treat the head as unsharded during load.

**The other TP>1 refusal.** The plugin also refuses EXL3 indexer shards under TP>1. The 2.15 pack's qkv is EXL3,
so it needs `0002`.

**Applying.** `./m31.sh build` runs `scripts/prepare_src.sh` on rank 0, which fetches the pinned plugin commit by its
full sha, applies the series with `git apply --check` first, verifies `_lm_head_full_vocab` is present, and
labels the image with the pins and the series sha256. `./m31.sh start` refuses an image whose labels do not match.

**Attribution.** Derived from tonyd2wild's `tp2-minimax-lmhead.patch` in
[MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark)
(AGPL-3.0-only). The commit message in the patch records that.

## `0002-minimax-exl3-indexer-tp2.patch`

**Target:** the plugin at `053c634` with `0001` applied (it also applies to the clean pin).
**License:** AGPL-3.0-only.
**Source:** tonyd2wild's `patches/tp2-minimax-indexer.patch` from
[MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark)
at `70c4e43`. It is copied byte for byte (sha256 `8af3e8600f0ef9856a93b77ce4d6ba5e520ac8c58a456d66b6b94f68c89b10ac`,
checked by `tests/test_recipe.sh`), and its mailbox header keeps his authorship.

**Problem.** MiniMax fuses its sparse-attention indexer (`index_q`, `index_k`, MXFP8) into the QKV GEMM. With an
EXL3 qkv, the plugin refuses TP>1:

```
EXL3 MiniMax indexer shards are not supported with TP size > 1
```

**Fix.** vLLM shards `index_q` like the KV heads and replicates `index_k` (one head). The patch cuts each rank's rows
of the MXFP8 payload and its E8M0 scales by that rule (`_indexer_rows_for_rank`), the same rule the plugin already
uses for the bf16 QKV weights.

## `0003-minimax-exl3-moe-prescan-tp-offset.patch`

**License:** AGPL-3.0-only. Ours.

**Problem.** The plugin plans each MoE layer's trellis arenas from a header-only prescan of the checkpoint, keyed by
global expert id `starting_expert_offset + local id`. It set that offset to `ep.rank * num_experts` whenever vLLM had
an EP group, and vLLM builds one (size 2) under TP=2 even without expert parallelism. Rank 1 therefore looked up experts
128-255 of a 128-expert model, the prescan returned `None`, and every MoE layer on rank 1 fell back to stage-and-pack:
21,888 `staging fallback` warnings per boot and about 60 s of arena packing in post-processing.

**Fix.** The offset is 0 when `layer.use_ep` is `False`, since every rank holds every expert under plain TP. Rank 1 then
logs 57 `PRESCAN` lines and no fallback, like rank 0. The weight snapshots (0004) rely on it: a staged load parks
tensors in dictionaries that only post-processing turns into arenas.

## `0004-minimax-exl3-weight-snapshots.patch`

**License:** AGPL-3.0-only. Ours; the idea follows the processed-weight snapshots of
[kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10) and the author's earlier DGX
Spark recipe, with no code taken from either.

**Problem.** Loading this pack takes 120-150 s per rank, and almost none of it is bytes. Raw copies run at about
27 ms/GiB on GB10. The cost is per-tensor Python: about 90,000 tensors, each name-mapped, TP-narrowed and copied.

**Fix.** A new module, `vllm_exl3/weight_snapshot.py`, wraps `DefaultModelLoader.load_weights` when
`VLLM_EXL3_WEIGHT_SNAPSHOT=1` (`m31.sh` default via `WEIGHT_SNAPSHOT=1`):
- **First boot of a configuration:**
  - the normal load runs;
  - every CUDA/CPU storage behind the model's parameters and buffers goes to one O_DIRECT file per rank, with a CRC32
    per 64 MiB chunk. That covers parameters the load re-points with `param.data = ...` and tensors it parks in module
    attributes (the MiniMax indexer payloads);
  - the manifest records the tensor layout and the plain module attributes the load set.
- **Later boots:**
  - the arenas are planned exactly as the loader plans them;
  - the layout must equal the manifest's;
  - the storages are read back on 4 threads through pinned buffers, and the attributes are replayed;
  - `process_weights_after_loading` then runs on identical inputs.
- **Fail-closed:**
  - a missing or mismatched snapshot loads the checkpoint, and so does a load whose side effects cannot be replayed
    (no snapshot is written);
  - a CRC failure deletes the snapshot and loads the checkpoint;
  - a failure while applying a restore deletes the snapshot and fails that boot, so the next boot loads cleanly.
- **Key:** checkpoint files (shards by size and mtime, JSON by content), plugin sources, torch/vLLM/exllamav3,
  every `VLLM_EXL3_*`/`EXL3_*` variable, TP rank and size, device, and `VLLM_EXL3_WEIGHT_SNAPSHOT_TAG`. One snapshot is
  kept per rank.

**Measured** (2026-10-24, this exact tree, both test pairs):
- write: 55.11 GiB per rank in 7.3-19 s;
- restore: 8.9-9.3 s at 6.3-6.7 GB/s;
- model load: from 150-160 s to 38 s;
- `start` to READY: 116.5-120.1 s;
- NLL check: within 1.3% of each pair's reference.

Disk: about 55 GiB per rank under `CACHE_DIR/weight-snapshots`.

## `0005-minimax-exl3-fat-tier-swiglu-oai.patch`

**License:** AGPL-3.0-only. Ours.

**Problem.** `apply_exl3_batched_fat` (experts with more than `VLLM_EXL3_FAT_THRESHOLD` routed rows) always applied
SiLU. MiniMax uses SwiGLU-OAI, `gate*sigmoid(1.702*gate)*(up+1)`. Short gates never route enough rows to reach the
tier. With 16K prefill chunks, 16K needles went 0/3 and long NLL rose to 11.15 (reference 7.26).

**Fix.** The tier computes the layer's activation, clamp included, and raises for activations it does not implement.

## `0006-minimax-exl3-lm-head-rank-blocks.patch`

**License:** AGPL-3.0-only. Ours.

**Fix.** Under TP>1 each rank computes only the 128-column LM head blocks covering its vocab partition, not all
200,064 logits. The LM head share of a decode step roughly halves.

## `0007-minimax-exl3-fat-gemm-any-k.patch`

**License:** AGPL-3.0-only. Ours; extends the plugin's native fat GEMM kernel.

**Fix.** The fused dequant+MMA fat GEMM is templated on trellis width K 1..8 and on the codebook (3INST, MCG, MUL1).
Mixed-K fat experts no longer fall back to reconstruct + cuBLAS + index_add.

**Measured:**
- equivalence with reconstruct + hgemm + Hadamard: 2e-4 relative on K2-5 × 3 codebooks;
- prefill: 884 to 921-926 tok/s;
- 16K needles 3/3; long NLL is deterministic at 6.92.

At M=16K dense shapes the fused kernel is slower than reconstruct + cuBLAS (46 vs 67 TFLOPS), so dense linears keep the
reconstruct path.

## `0008-minimax-exl3-grouped-fat-fp8-prefill.patch`

**License:** AGPL-3.0-only. Ours; the grouped GEMM extends the MIT fat GEMM kernel of 0007.

**Grouped fat tier** (default; `VLLM_EXL3_FAT_GROUPED=0` restores the per-expert loop):
- one fused dequant+MMA launch per (K, codebook) class per projection for every fat expert of a layer, using
  per-expert row bounds and trellis/svh pointer tables;
- 64-wide K stages with register double-buffering;
- the row gather fused into the input Hadamard, and clamp + activation + fp16 round fused into the down-input
  Hadamard.

The per-expert loop issued about 13 launches per fat expert, and its small per-expert grids left most SMs idle.

**Measured** (one test pair, dev build, 8K batches, threshold 256):
- gate shape (128 experts, 32K rows, 6144→1536, K3): 49.7 → 12.9 ms; outputs equal the per-expert kernels;
- prefill 8K/16K: 1006/1038 tok/s against 859/876 for 0007;
- 16K needles 3/3; long NLL 6.93-6.98.

**FP8 dense prefill** (`VLLM_EXL3_FP8_PREFILL=1`, off by default):
- e4m3 in LinearEXL3's rotated basis: per-row quantized `had(x*suh)`, a per-tensor FP8 copy of the bare trellis
  weight, a tensorwise `torch._scaled_mm`, then the row/weight rescale and output Hadamard written into each shard's
  slice of one bf16 output. Rowwise-scaled `_scaled_mm` selects a ~53 TFLOPS kernel on GB10 at N ≥ 6144;
  tensorwise runs at 150-170.
- 1.56-1.62× per dense linear, +7.5% prefill.
- Long NLL rises 3-6% (synthetic 6.95 → 7.38; natural docs and code 4.87 → 5.05). With attention excluded
  (`VLLM_EXL3_FP8_PREFILL_SKIP=self_attn`) the cost stays (4.97-5.13). The cause is e4m3's 3-bit mantissa on both
  operands. The NLL check scores short passages that never reach the 1,024-row FP8 cutoff, so it cannot
  see this cost.

## `0009-minimax-exl3-bf16-dense-prefill.patch`

**License:** AGPL-3.0-only. Ours.

**Problem.** For prefill-sized batches, every dense EXL3 layer (attention qkv/o, shared experts, the three dense MLPs)
copied the bf16 activations to fp16, ran reconstruct + `hgemm` per shard into fp32 outputs, concatenated the shards
and cast back to bf16. Those copies and casts took ~1.5 s of a 15.4K-token prefill.

**Fix.** `Exl3LinearMethod._apply_wide_prefill`, for bf16 activations with ≥ `VLLM_EXL3_BF16_PREFILL_MIN_ROWS` rows
(1,024):
- each run of consecutive EXL3 shards is reconstructed in the original basis (`reconstruct_had_slice`, both
  Hadamards folded, as LinearEXL3's fused reconstruct does);
- the run is rounded to bf16 into one `[K, run N]` weight and multiplied in one bf16 GEMM (fp32 accumulate) straight
  into its column slice of the layer's bf16 output;
- indexer (MXFP8) and BF16 shards are copied into their slices.

The FP8 path from 0008 shares this layer plan and takes precedence when enabled. `VLLM_EXL3_BF16_PREFILL=0` restores
the fp16 path. The env knobs are apply-only in the snapshot key.

**Numerics.** The only change is bf16 rounding of the reconstructed weights: 2.6e-3 relative per GEMM output against
the fp16 path. For comparison, the fat tier is ~1e-3 and FP8 3.7e-2. BF16-shard columns are bit-identical.

**Measured** (one test pair, dev build):
- per layer at M=8192: 1.10-2.0× (qkv-like 1.47×);
- prefill 8K/16K: 1004/1038 → 1074/1103 tok/s;
- needles 3/3; NLL check ≤ 1.1%; chat smoke correct.

## `0010-minimax-exl3-moe-prefill-kernels.patch`

**License:** AGPL-3.0-only. Ours; edits the MIT-derived grouped fat GEMM from 0008.

**Mixed-K MoE row tile.** The plugin always passed `m_tile=16` to `exl3_moe_mixedk`. exllamav3 also instantiates
32/64-row tiles for the mul1 codebook, which this pack uses. With a 64-row tile each dequantized weight tile serves up
to 64 rows; the kernel drops to 32/16-row tiles per expert as rows run out. Batches with ≥ 2048 tokens (not during
graph capture) now pass 64; `VLLM_EXL3_MIXEDK_M64_MIN_TOKENS` sets the cutoff and 0 disables it.

Measured on the per-rank shapes:

| Batch tokens | m_tile 16 | m_tile 64 |
|---|---|---|
| 1,024 | 17.6 ms | 18.9 ms |
| 2,048 | 33.5 ms | 28.5 ms |
| 8,192 | 47.0 ms | 35.6 ms |

In the served 15K prefill the mixedk time went 2.63 → 2.34 s. The output is within 7e-4 of the 16-row tile, and both
are ~6.5e-4 from an fp32 reference (`tests/gpu/bench_mixedk_prefill.py`).

**Grouped fat GEMM occupancy.** The kernel used 131 registers, so only one 256-thread CTA fit per SM. Changes:
- `__launch_bounds__(256, 2)` brings it to 128 registers with no spill, so two CTAs fit per SM;
- the epilogue stores float4, and the scatter uses `red.global.add.v4.f32`.

Serving-like fat loads (24 experts, 10K rows): gate 3.67 → 3.25 ms, down-scatter 5.68 → 4.38 ms. Outputs are
bit-equal to the per-expert kernels.

**Screen** (one test pair, dev build): prefill 8K/16K 1074/1103 → 1120/1161 tok/s; needles 3/3; NLL check ≤ 0.5%, plus the
known bimodal dialogue flip; chat smoke correct.

## `0011-minimax-exl3-mixedk-coop-decode.patch`

**License:** AGPL-3.0-only. Ours.

**Problem.** exllamav3's decode MoE kernel `exl3_moe_coop` takes one (Kg, Ku, Kd) per launch and an expert range
`[lo, hi)`. It indexes its pointer tables by `expert - lo`. Mixed-K packs therefore always ran `exl3_moe_mixedk`, a
general stream-K GEMM. At decode that kernel is instruction-bound: Nsight Compute shows 12.4 thread-instructions per
dequantized weight and IPC 1.85.

**Fix.**
- At load: for a mixed-K layer with ≤ 4 K groups (gated, Kg == Ku), experts are renumbered into K-tuple order. This
  produces per-group pointer-table slices and an original → ordered id map.
- At decode: batches of ≤ `VLLM_EXL3_COOP_SPLIT_MAX_TOKENS` tokens (24 by default, the spec-verify rows of 6 requests)
  map router ids through the map, so routing is unchanged, and run one coop launch per group.
- Each launch adds the previous launch's output through coop's `sh_out`, so every row is written once per launch.
- `VLLM_EXL3_COOP_SPLIT=0` keeps mixedk.

**Numerics.** On live weights, coop differs from mixedk by 1e-3 per layer. Against an fp32 reference, coop is the
closer of the two: 6.4-6.5e-4 vs 6.6e-4, with skewed routing up to 48 tokens.

Without the 24-token cap, the short NLL-check passages (32-48 tokens) took coop, and the reasoning passage read 18.8%
off its mixedk-pinned reference, repeatably. The probe reacts that strongly to ~1e-3 perturbations. The cap keeps
prefill-sized batches on mixedk.

**Measured:**
- kernel time: 1 token 323 → 190 µs, 4 tokens 885 → 653 µs;
- served decode c=1 23.5 → 24.5 tok/s, c=4 52.5 → 59.3 tok/s (B screen);
- NLL check ≤ 1.1%; needles 3/3; chat smoke correct.

## `0012-minimax-exl3-tiled-sparse-attention.patch`

**License:** AGPL-3.0-only. Ours; a new Triton kernel next to vLLM's `_gqa_sparse_fwd_kernel`.

**Problem.** vLLM's prefill sparse attention runs one program per (query token, KV head). Each program loads that
token's 16 selected 128-token blocks, so every block is re-read from L2 by every token that selected it. It took
1.5 s of a 15K prefill.

**Fix.** `sparse_attn_tiled.py` runs one program per 8 consecutive rows of one request:
- it iterates the sorted union of their selected blocks once;
- it masks per row for selection, causality and sequence length, so every row attends exactly its own blocks.

Captured live selections are strongly correlated: an 8-token tile's union averages 18.9 blocks, against 128 if
nothing were shared.

The q4 prefill wrapper uses it for calls with ≥ `VLLM_EXL3_SPARSE_TILED_MIN_ROWS` rows (1024 by default); shorter
calls keep vLLM's kernel. `VLLM_EXL3_SPARSE_TILED=0` disables it.

**Numerics.** Only the order in which blocks enter the online softmax changes. Against an fp32 reference it matches
vLLM's kernel to three digits (1.5-2.3e-3 for both, `tests/gpu/test_sparse_attn_tiled.py`). Without the row cutoff,
the short NLL-check passages moved 7-17%, the same short-passage probe sensitivity as in 0011.

**Measured:**
- per call (8K rows): 12.4 → 5.6 ms;
- served prefill 8K/16K 1167/1221 → 1237/1268 tok/s;
- needles 3/3; NLL check ≤ 2.5%, plus the known dialogue flip.

## `0013-minimax-exl3-fat-prefill-threshold.patch`

**License:** AGPL-3.0-only. Ours.

**Fat-tier prefill threshold.** Batches of at least `VLLM_EXL3_FAT_PREFILL_TOKENS` tokens (default 1024, i.e.
prefill chunks) send experts above `VLLM_EXL3_FAT_THRESHOLD_PREFILL` rows to the grouped fat GEMM. The default for
that threshold is `VLLM_EXL3_FAT_THRESHOLD`. Shorter batches keep `VLLM_EXL3_FAT_THRESHOLD`, which also caps coop
decode eligibility, so their numerics and kernel choice do not change. The q4-300k profile uses 128 / 64 / 1024.

Two single-threshold variants were rejected:
- Threshold 64 everywhere moved the NLL-check code passage.
- Threshold 64 with the fat tier disabled below 1024 tokens kept short batches out of the coop path, costing c=4
  decode ~10% (one test pair, same session: 62.2 vs 68.9 tok/s).

**Tiled attention JIT.** The kernel no longer specializes on the tile count or the block-table stride. Both change
every call, and divisibility specialization compiled a new variant per shape class.

## `0014-minimax-exl3-tiled-attention-no-sync.patch`

**License:** AGPL-3.0-only. Ours.

**Problem.** The tiled attention wrapper read `cu_seqlens_q` and `prefix_lens` with `.tolist()` on every layer and
built tiles in a Python loop. Each read synchronized the stream, serializing kernel launch and execution 60 times
per prefill chunk. On the busier of the two test pairs, the benchmark's first 8K prefill ran ~10% below a direct measurement.

**Fix.**
- The tile count is bounded from shapes alone: `total_q // tile + num_reqs`.
- The tile → request map is computed on the GPU with `searchsorted`.
- Surplus tiles carry zero rows and exit immediately.

Outputs are unchanged.

## `0015-minimax-exl3-fat-grouped-cp-async.patch`

**License:** AGPL-3.0-only. Ours.

**Grouped fat GEMM pipeline.** The register prefetch is replaced by a `cp.async` pipeline: 5 stages × 32 K for
trellis widths up to 3 bits, 4 stages otherwise. This frees 32 registers and cuts shared-load waits from 3.64 to
0.85 per issued instruction.

Other changes:
- The stage barrier sits at the top of the last sub-block.
- M-blocks past an expert's last row skip loads and MMAs.
- The epilogue handles 32 rows per round with a padded stride (136), issues routing loads early, and has one barrier
  fewer per round.

The old kernel is removed; the new one covers trellis widths K 1-8 and all codebooks. Results are bit-identical
(`tests/gpu/test_fat_gemm_grouped.py`; accumulation order unchanged).

Measured on GB10 at the live shapes (125 experts, 33,173 rows):

| Shape | Before | After |
|---|---|---|
| gate\|up K2 | 21.3 ms | 17.7 ms (−17%) |
| down K2 (scatter) | 14.1 ms | 13.1 ms (−7%) |
| gate\|up K3 | | −16% |
| down K3 | | −11% |

Diagnosis: removing the dequant saves 2.6 ms and the A `ldmatrix` loads only 1.2 ms, so a 2×4 warp layout was not
pursued. A 1×4 layout was slower on K2.

## `0016-minimax-exl3-skip-profile-compute.patch`

**License:** AGPL-3.0-only. Ours.

**Profile-run skip.** When `kv_cache_memory_bytes` is set (this recipe always sets `KVB`), vLLM still runs
`profile_run`. That is a full `max_num_batched_tokens` forward on dummy tokens, without attention metadata, used
only as warm-up: no memory figure comes from it and its outputs are discarded. The expert and wide dense GEMMs now
return zeros of the right shape there (`VLLM_EXL3_SKIP_PROFILE_COMPUTE`, default on).

Without `kv_cache_memory_bytes`, the profile sizes the KV cache and runs in full. CUDA-graph capture and real batches
(which carry attention metadata) never skip.

Measured on B with 16K batches:
- init engine 18.1 → 11.5 s;
- the first 8K/16K prefill after boot runs at warm speed (1427/1431 tok/s);
- NLL check 0.28%; needles 3/3.

## `0017-minimax-exl3-fat-grouped-full-tile-loop.patch`

**License:** AGPL-3.0-only. Ours.

**Fat GEMM full-tile loop.** In 0015, the per-block exit for rows past an expert's end sat inside the unrolled M-block
loop. That pinned each `ldmatrix` directly before its MMA, and short-scoreboard stalls hit every HMMA (22% of
samples on gate K2). Full tiles now run a loop without exits; an expert's partial last tile keeps the exit. The MMAs
and their order are unchanged, so results are bit-identical.

Measured on GB10 at the live shapes:
- gate|up K2: −3.1%;
- gate|up K3: −4.5%;
- down: within noise.

256-row tiles (decoding B once per 256 rows) were 18-33% slower: one CTA per SM costs more than the halved dequant
saves.

## `0018-minimax-exl3-coop-mixedk-single-launch.patch`

**License:** AGPL-3.0-only. Ours.

**Mixed-K decode on one coop launch per stage.** Patch 0011 issued one coop launch per K group. A mixed-K layer now
runs one `exl3_moe_coop_mk` launch per stage (rot, gate/up, down), with per-expert K read from the layer's K tables.
`VLLM_EXL3_COOP_MK=0` restores the per-group path. Requires exllamav3 patch 0002.

## `0019-minimax-exl3-fat-prefill-glue-fusion.patch`

**License:** AGPL-3.0-only. Ours.

**Prefill MoE glue fusion** for the fat tier:
- `exl3_had_r_128_token_gate_up` reads each token's `x` once (bf16 or fp16) and writes the pre-scaled,
  Hadamard-transformed gate and up inputs for all of that token's fat routes. It optionally zeroes the fp32 routed
  accumulator in the same pass. It replaces `x.half()`, the zero fill and two gathered Hadamard launches.
- `exl3_fat_gemm_grouped_act` adds an `ACT` epilogue to the grouped up GEMM: clamp, SwiGLU-OAI or SiLU, fp16 round,
  down-suh scale and Hadamard, writing the down input directly. It replaces `act_had_r_128_grouped`.
- When every routed expert is fat (q4-300k: `FAT_THRESHOLD_PREFILL=0`), the layer skips the standard mixed-K path
  entirely, with one host read of the counts.

Intermediates are bit-identical (`tests/gpu/test_fat_gemm_grouped.py`). The layer output differs by ≤ 2.4e-7, from
atomic order only. On real 16K-chunk routing: 78.9 → 72.5 ms per MoE layer.

## exllamav3 patches (`patches/exllamav3/`, `EXL3_PATCH_FILES`)

**License:** MIT (exllamav3). Ours. Applied in order to the pinned exllamav3 commit `EXL3_REV` (`fdd77361`,
vcruz305's fork with the act-3 MoE fix).

- `0001-exl3-coop-decode-prefetch-dsplit.patch`: coop MoE decode with an unpredicated slice loop, an A-fragment
  ring, weight prefetch ahead of setup, and device-side split-k for launches under three waves (fp32 partials for
  the whole launch when any item splits). Unsplit launches are bit-identical. Split launches differ by ≤ 6e-4 rel;
  accuracy against fp32 is 6.44e-4 (base 6.52e-4). `EXL3_MOE_COOP_DSPLIT=1` disables the split.
- `0002-exl3-coop-decode-mixedk-instances.patch`: kernels templated on a K range, plus mixed instances ([1,2],
  [2,3], [3,4]) that read each expert's K from device tables. A new binding, `exl3_moe_coop_mk`, takes them.
  Uniform instances are unchanged; mixed vs uniform is bit-identical on uniform layers.
- `0003-exl3-dense-decode-split-k-gemv.patch`: `exl3_dgemv`, a split-k decode GEMV. It makes one regular launch
  with no grid barrier, uses fp32 MMA accumulation, and the last block to arrive reduces in split order. `exl3_gemm`
  takes it for weights ≤ 8 MB at ≤ 8 rows or ≤ 2 MB at 9-16 rows: k/v projections and the shared expert.
  - Speed, 4 rows: k/v 21.6 → 11.2 µs; shared gate/up 30.2 → 25 µs.
  - Accuracy against fp32 is unchanged (2.1e-4); on shared gate/up it improves from 3.9e-4 (the old fp16-accumulating
    GEMV).
  - `EXL3_DGEMV=0` disables it.
  - The vLLM router-gate kernel swap measured alongside was not adopted: the router is off-limits.

Live routing on GB10, coop time per layer:

| Patch | c=1 | c=4 |
|---|---|---|
| 0001 | 445 → 414 µs | 1045 → 1010 µs |
| 0002 (on 0001), 45 layers | −6.3% | −3.8% |
| 0002 (on 0001), 12 layers | −8.1% | −10.2% |

Live server profile (one test pair, 1 stream): 0001 alone takes coop time from 0.451 to 0.426 ms per layer.

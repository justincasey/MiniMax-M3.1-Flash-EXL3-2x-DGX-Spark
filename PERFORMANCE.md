# Performance

The serving stack is the image built from this release: `vllm-exl3` `053c634d` + `patches/0001`-`0019`,
`exllamav3` `fdd77361` + `patches/exllamav3/0001`-`0003`, on the pinned vLLM base image. It ran on two
pairs of DGX Sparks (GB10), each pair joined by one QSFP cable at TP=2.

**Status:** these are benchmark runs, not a committed `bench/records/` run. Every profile is still marked
`PROFILE_MEASURED=0`.

## Headline (`q4-300k`, both pairs)

Profile: 300K context × 6 streams, q4 (nvfp4) KV cache, DSpark k=3, 16K prefill chunks.

| Metric | Mean | Pair A | Pair B |
|---|---|---|---|
| Prefill, 8K prompt (tok/s) | 1,559 | 1,565 | 1,554 |
| Prefill, 16K prompt (tok/s) | 1,563 | 1,565 | 1,561 |
| **Prefill, mean (tok/s)** | **1,561** | 1,565 | 1,557 |
| **Decode, 1 stream (tok/s)** | **30.8** | 30.7 | 31.0 |
| **Decode, 4 streams total (tok/s)** | **66.4** | 67.6 | 65.2 |
| Boot to ready (s) | 104.3 | 100.4 | 108.2 |
| Free memory at ready (GiB, lowest rank) | 25.8 | | |

Decode by prompt type:

| Type | 1 stream | 4 streams (total) |
|---|---|---|
| code | 34.7 | 76.6 |
| structured | 31.6 | 62.0 |
| json | 30.1 | 69.2 |
| prose | 26.9 | 57.7 |

Speculative decoding preserves the target model's output distribution; how many draft tokens are accepted depends
on the text, which is why code decodes faster than prose.

Quality gates, all passing:
- NLL drift against a pinned reference: 1.4% (the gate is 10%);
- 16K needle at 10%, 50% and 90% depth: 3 of 3;
- chat smoke test;
- memory guard: no kills.

## Release validation (fresh clone, both pairs)

This release was cloned fresh onto each pair and taken through `doctor`, `fetch`/`verify`, `build`, `start` and every
gate below. That clone carried the two engine commits as `git archive` tarballs; the build context staged from them
is byte-identical, file contents and modes, to the one `build` now stages from the pinned commits on GitHub.

| Check | Pair A | Pair B |
|---|---|---|
| Prefill, mean of 8K and 16K (tok/s) | 1,571 | 1,578 |
| Decode, 1 stream / 4 streams total (tok/s) | 30.6 / 66.3 | 31.3 / 67.5 |
| First boot (writes the weight snapshot) / snapshot boot (s) | 226 / 100 | 224 / 106 |
| NLL drift against the earlier stack's reference (largest passage) | 2.03% | 0.53% |
| 16K needle, 3 depths | 3 of 3 | 3 of 3 |
| `quality` | 50 of 50 | 50 of 50 |
| `tools` | 24 of 27 | 24 of 27 |
| `api-test` | no failures | no failures |

The three `tools` misses are the same on both pairs. Asked to create an order, the model first looks the customer
or SKU up with `run_sql`, and the single-turn check only accepts `create_order` as the first call. `api-test`
sometimes warns that two temperature-0 replies differ; greedy output is not bit-reproducible here.

Two checkers were fixed during this run, and the `tools` and `api-test` rows come from the rerun with the fixes,
which this release includes. Before them, `tools` scored 23 of 27 (it rejected a correct call whose list argument
had an extra entry) and `api-test` failed `/tokenize` on every install (it called the endpoint under `/v1`, where
vLLM doesn't serve it).

## Context profiles

Each profile was booted on one pair and checked for:
- concurrent decode at its stream count, with 0 errors;
- 64K needles at 10% and 90% depth: 2 of 2;
- the NLL check within the 10% gate;
- the memory guard's lowest free-memory reading.

| Profile | Boot to ready (s) | Decode total, prose / code (tok/s) | Free at ready, rank 0 (GiB) | Lowest during tests (GiB) | NLL drift |
|---|---|---|---|---|---|
| `q4-215-256k-2s` | 95.5 | 38.3 / 48.5 | 31.7 | 27.2 | 6.88% |
| `q4-215-256k-4s` | 93.5 | 62.4 / 80.8 | 17.9 | 13.7 | 1.34% |
| `q4-215-256k-8s` | 98.5 | 55.8 / 72.8 | 16.5 | 12.1 | 1.30% |
| `q4-215-512k-2s` | 100.3 | 39.8 / 51.0 | 18.3 | 13.7 | 1.48% |
| `q4-215-512k-4s` | 100.1 | 62.3 / 75.6 | 17.1 | 12.7 | 0.80% |
| `q4-215-512k-8s` | 96.5 | 53.7 / 70.1 | 17.4 | 13.0 | 1.03% |

The 6.88% drift is one short check passage that flips between about 0% and 6.88% from run to run; it is within the
10% gate.

At 8 streams the total decode rate is below the same context length at 4 streams (55.8 / 72.8 against
62.4 / 80.8 at 256K); the 8-stream profiles buy concurrency, not throughput.

**Full-context runs** (a probe with a 32 GiB pool and 8 streams, a harder memory case than any shipped profile):

| Prompt | Needle at 50% depth | Time to first token | Lowest free memory |
|---|---|---|---|
| 251,900 tokens | pass | 14 min (~305 tok/s) | 9.1 GiB |
| 502,251 tokens | pass | 36 min (~230 tok/s) | 7.7 GiB |

Attention cost grows with context length, so prefill speed falls as prompts get longer.

## Method

- **Prefill:** one cold request per length, each with a unique nonce so no prefix cache applies. Time to the first
  token, with the token count from the engine's `usage.prompt_tokens`.
- **Decode:** fixed 30-second windows of 256-token requests at temperature 0, for prose, code, structured and json
  prompts. Tokens come from the engine's `usage.completion_tokens`, never from streamed-chunk counts. The "total"
  numbers are aggregate throughput across all streams.
- **NLL check:** teacher-forced mean NLL of six fixed passages, compared with a reference pinned on the same stack.
- **Commands:** `./m31.sh bench decode | prefill | needle | check`; see `bench/README.md`.

## Known limits

- **Tested hardware:** two pairs of Sparks on one site's RoCE fabric.
- **Not exercised:** all streams at full context at the same time (hours of prefill), and multi-hour soak runs.
- **Boot time varies by pair:** 100 s on one and 108 s on the other in the same run.
- **Transient first boot:** in earlier runs the first boot of a new image failed once and succeeded on retry. It
  did not happen in the fresh-clone validation on either pair. `m31.sh` saves both ranks' full logs when a start
  fails.
- **Prefill tuning used synthetic prompts.** In a 16K chunk of those prompts one expert receives about half the
  tokens. The tuning was not repeated on natural-text routing.
- **Ceilings at TP=2:**
  - Prefill is near an estimated ~1,600 tok/s ceiling (bf16 GEMMs plus sparse attention and allreduce that don't
    overlap with compute).
  - Decode is bound by memory bandwidth, with the main kernels at 180-230 GB/s against a practical ~250 GB/s.

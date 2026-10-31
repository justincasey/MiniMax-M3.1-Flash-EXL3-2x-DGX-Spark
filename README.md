# MiniMax-M3.1-Flash EXL3 2.15 bpw at TP=2 on two DGX Sparks

A serving recipe for [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash). It serves
vcruz305's [EXL3 SAGE 2.15 bpw pack](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw) with
tensor parallelism across **two NVIDIA DGX Sparks** (GB10) joined by one QSFP cable. The endpoint is an
OpenAI-compatible vLLM server, and decoding uses the pack's bundled DSpark speculative decoder.

**This repository contains no model weights.** It holds the controller script, the build pins (the engine commits
and two patch series), the serving profiles and a benchmark harness. The weights are downloaded from the pack's
Hugging Face repo at a pinned revision, and every file is checked against a sha256 manifest.

## Performance

These are benchmark runs, not a published `bench/records/` record. They were measured on two
independent pairs of Sparks with the `q4-300k` profile: 300K context, 6 streams, q4 KV cache, DSpark k=3, 16K
prefill chunks.

| Metric | Result | Pair A | Pair B |
|---|---|---|---|
| Prefill, mean of 8K and 16K prompts (tok/s) | **1,561** | 1,565 | 1,557 |
| Decode, 1 stream (tok/s) | **30.8** | 30.7 | 31.0 |
| Decode, 4 streams total (tok/s) | **66.4** | 67.6 | 65.2 |
| Boot to ready | 104 s | 100 s | 108 s |
| NLL drift vs reference / 16K needles | 1.4% / 3 of 3 | | |

Decode speed depends on how often the draft is accepted, so it varies with the text: at 1 stream, code runs at 34.7
tok/s and prose at 26.9. [`PERFORMANCE.md`](PERFORMANCE.md) has the method, the per-type numbers, the context
profiles and the known limits.

## Quick start

Everything runs from a control machine (a Mac works) over ssh to the two Sparks. Check [What you need](#what-you-need)
and [Node setup](#node-setup) first; `doctor` tells you what is missing.

```bash
git clone https://github.com/justincasey/MiniMax-M3.1-Flash-EXL3-2x-DGX-Spark m31
cd m31

cp .env.example .env          # set NODES (rank 0 first)
./m31.sh doctor               # checks both nodes (installs helper scripts under ~/m31), finds the QSFP link
./m31.sh fetch                # pinned pack: split download + cable fan-out + sha256 of all 40 files
./m31.sh build                # engine image on rank 0 from the pinned commits, shipped over the cable
./m31.sh start                # guards armed, both ranks launched; waits for /v1/models, warm-up, smoke
./m31.sh probe                # capability + metrics probe before trusting a benchmark
```

What to expect:
- **`fetch`** splits the download: each node fetches about half (~55 GiB), then the halves are exchanged over
  the cable, so each node ends with the full 110.8 GiB pack.
- **`build`** takes 7-20 min on rank 0 and needs about 60 GiB of `MemAvailable` there.
- **The first `start`** takes about 4 minutes, because it loads the checkpoint and writes a per-rank weight
  snapshot (56 GiB on disk per node). Later starts restore that snapshot and are ready in about 100 s.

## What is pinned

Everything that goes into the served stack is fixed in [`config/pins.sh`](config/pins.sh) and the lane file
[`config/lanes/2.15bpw.sh`](config/lanes/2.15bpw.sh):

| Input | Pin | Source |
|---|---|---|
| Base image | `vllm/vllm-openai:minimax-m31-flash` @ `sha256:86f582da…` (multi-arch index; arm64 `sha256:5d763476…`) | Docker Hub |
| exllamav3 | [`vcruz305/exllamav3`](https://github.com/vcruz305/exllamav3) @ `fdd77361` (vcruz305's fork of turboderp-org/exllamav3 with the act-3 MoE fix) | GitHub |
| vllm-exl3 | [`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3) @ `053c634d` | GitHub |
| exllamav3 patches | `patches/exllamav3/0001`-`0003` | this repo |
| Plugin patches | `patches/0001`-`0019` | this repo |
| Weights | `vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw` @ `99bf3a57`; `config/lanes/2.15bpw/pack-manifest.tsv` (40 files, sha256 + size, 118,985,525,140 bytes) | Hugging Face |

**How the build uses the pins:**
- **Engine sources.** `build` fetches each pinned commit by its full sha on rank 0 (a full sha names exactly one
  source tree), exports it with `git archive`, and applies its patch series fail-closed. The exports carry the upstream
  LICENSE and NOTICE files. The Docker build itself downloads nothing.
- **Image identity.** The image tag and labels carry the plugin revision and one sha256 over both patch series.
  `start` refuses an image whose labels don't match the pins.
- **Weights.** `fetch` downloads the pinned revision and checks every file's sha256 against the manifest.

The engine commits and the base image live outside this repo. If a pinned commit or the base digest ever
disappears upstream, the build stops rather than building something else.

## What the pack is

The whole pack is 110.81 GiB in 40 files. At TP=2 each rank holds about 55 GiB of it (the weight snapshot is 55.1
GiB), and vLLM reports 58.9 GiB of GPU memory per rank once the model is loaded:
- **EXL3 (trellis, per-expert K):** the routed experts, 57 MoE layers × 128 experts with top-4 routing, plus every
  other linear (421 modules).
- **MXFP8:** the sparse-attention indexer, fused inside the EXL3 QKV.
- **bf16:** the embedding, 2.29 GiB.
- **DSpark draft:** 5 layers.

The pack's `config.json` is complete, so no quantization overlay is needed. The census is in
`config/lanes/2.15bpw/pack-census.json`.

The model itself:
- 428B parameters in total, 23B active per token;
- 60 layers: 3 dense and 57 MoE;
- 64 attention heads and 4 KV heads, with MSA sparse attention on every layer;
- a 1,048,576-token position window.

The pack has **no vision-tower weights**: this recipe is text only.

## What you need

**Hardware**
- **Two DGX Sparks** (GB10, sm_121) on DGX OS with the stock 4 KiB-page kernel, with nothing else on their GPUs.
- **One QSFP cable** between them (see [Node setup](#node-setup)).
- **≥ 230 GiB free disk per node** (`doctor` enforces it). Measured use is ~196 GiB: pack 111 GiB, weight
  snapshot 56 GiB, image 27 GiB, caches ~1 GiB. The rest is room for build layers and an older image during
  an upgrade.

**Each Spark**
- Docker with GPU access (DGX OS ships it);
- passwordless sudo for the ssh user, which the memory guard and sysctls need;
- `rsync`, `git` and `python3` 3.9+;
- on rank 0, https access to github.com (`build` fetches the two engine commits; `doctor` checks it);
- the Hugging Face CLI at `~/venvs/hf/bin/hf`, or set `HF_BIN`.

**The control machine**
- bash 4+ (on macOS: `brew install bash`);
- python3, curl, rsync and git;
- key-based ssh to both Sparks. The user defaults to `spark`; change it with `SSH_USER`.

## Node setup

Run on **each** Spark (as the ssh user):

```bash
# Hugging Face CLI where the recipe looks for it
python3 -m venv ~/venvs/hf && ~/venvs/hf/bin/pip install -U "huggingface_hub[cli]"

# Passwordless sudo for this user
echo "$USER ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/90-m31 && sudo chmod 440 /etc/sudoers.d/90-m31
```

**The QSFP link.**
- Give both ends of the cable addresses on a shared subnet, with MTU 9000. NVIDIA's DGX Spark documentation for
  connecting two Sparks covers this. On GB10 the cable shows up as two PCIe functions; the recipe uses both.
- `doctor` finds the link by itself: shared subnets, RoCE state and GID, MTU, and a ping across the cable, and it
  prints what it picked for each rank (cable address, interface, HCAs). If detection picks the wrong interfaces,
  pin all seven link keys in `.env`: `RANK0_IP`, `RANK1_IP`, `RANK0_HCA`, `RANK1_HCA`, `NCCL_IB_GID_INDEX`,
  `FABRIC_IF_0`, `FABRIC_IF_1` (all or none).

**Node-to-node ssh over the cable.** Rank 0 pushes the weights and the image to rank 1 this way. Without it the
recipe falls back to the management LAN, which is slower. On rank 0:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # skip if a key exists
ssh-copy-id <user>@<rank 1's cable address>
```

Then, from the control machine, `./m31.sh doctor` should end with `0 failure(s)`. Before `build`, the only
warnings are one per node saying the image is not built yet.

## Run

```bash
./m31.sh start        # memguard armed, rank 1 headless, rank 0 serves; waits for /v1/models, then warm-up + smoke
./m31.sh probe        # models, tokenize, chat, reasoning field, tool call, spec metrics, reasoning_effort
./m31.sh status
./m31.sh logs 0 100
./m31.sh stop         # saves each rank's log, stops guards, restores host sysctls
```

The API is OpenAI-compatible:

```bash
curl http://192.0.2.11:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "minimax-m3.1-flash",
  "messages": [{"role": "user", "content": "Is 221 prime?"}],
  "max_tokens": 4096
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.0.2.11:8000/v1", api_key="unused")
r = client.chat.completions.create(model="minimax-m3.1-flash", max_tokens=4096,
    messages=[{"role": "user", "content": "Explain MSA in one paragraph."}])
print(r.choices[0].message.content)
```

The chat template is **adaptive**: the model decides whether to think. Reasoning arrives separately as
`reasoning`/`reasoning_content` (this vLLM build uses `reasoning`) and counts toward `max_tokens`.

## Profiles

`PROFILE` picks a file in `config/profiles/`. The default is `q4-300k`. Every profile pins its KV pool (`KVB`) and
uses the tuned prefill settings.

| Profile | Context × streams | KV pool per rank | Full-length requests that fit at once | Decode, total, prose / code (tok/s) |
|---|---|---|---|---|
| `q4-300k` (default) | 300,000 × 6 | 18.6 GiB | 2 | 66.4 at 4 streams (all four prompt types) |
| `q4-215-256k-2s` | 262,144 × 2 | 14 GiB (546K tokens) | 2 | 38.3 / 48.5 |
| `q4-215-256k-4s` | 262,144 × 4 | 28 GiB (1.09M tokens) | 4 | 62.4 / 80.8 |
| `q4-215-256k-8s` | 262,144 × 8 | 28 GiB | 4 (~136K each if all 8 are long) | 55.8 / 72.8 |
| `q4-215-512k-2s` | 524,288 × 2 | 28 GiB | 2 | 39.8 / 51.0 |
| `q4-215-512k-4s` | 524,288 × 4 | 28 GiB | 2 (~273K each if all 4 are long) | 62.3 / 75.6 |
| `q4-215-512k-8s` | 524,288 × 8 | 28 GiB | 2 (~136K each if all 8 are long) | 53.7 / 70.1 |

What to know about the profiles:
- **The 28 GiB pool cap** is estimated to leave about 11.7 GiB free in the worst measured case, a full-context
  prefill. That estimate comes from a 32 GiB probe, which left 7.7 GiB, minus 4 GiB of pool. At 28 GiB, only
  64K-token prefills were run, with at least 12.1 GiB free. The measurements are in
  [`config/profiles/lane-2.15.inc`](config/profiles/lane-2.15.inc).
- **Eight streams decode less in total than four.** Their verify batch is 32 rows, past the coop decode path's
  24-token bound. The `-8s` profiles serve more clients at once, each at about half the speed.
- **When the pool fills,** vLLM preempts requests instead of running out of memory.
- **Full-context prompts are slow:** about 14 min to the first token at 252K tokens and 36 min at 502K.
- **All profiles are marked unmeasured** (`PROFILE_MEASURED=0`; `start` prints a warning) until a run is
  committed under `bench/records/`.

## Configuration

Precedence: caller environment > `.env` > `m31.sh` defaults > the profile. Pins cannot be changed from `.env`.

| Key | Default | Meaning |
|---|---|---|
| `NODES` | (required) | two ssh targets, rank 0 first |
| `PROFILE` | `q4-300k` | `config/profiles/<name>.env` |
| `MAX_MODEL_LEN` / `MAX_NUM_SEQS` | profile | request window and concurrent sequences |
| `KVB` | profile | `--kv-cache-memory-bytes`. A profile without it derives `MAX_MODEL_LEN × MAX_NUM_SEQS × KV_BYTES_PER_TOKEN × 1.02`, 256 MiB-aligned |
| `KV_BYTES_PER_TOKEN` | `28160` | derivation constant; the measured cost is 27,520 bytes/token/rank (q4 main KV, fp8 indexer, fp8 draft) |
| `KV_Q4` / `KV_CACHE_DTYPE` | `1` / `nvfp4` | q4 main-KV pool through `VLLM_EXL3_Q4KV`; requires the fp8 indexer and fp8 draft |
| `INDEXER_KV_DTYPE` / `INDEXER_MXFP8` | `fp8` / `1` | indexer cache and its on-device MXFP8 retention |
| `DSPARK` / `NUM_SPECULATIVE_TOKENS` | `1` / `3` | the draft ships inside the pack; k=3 verifies up to 4 tokens per step |
| `DRAFT_ATTN_BACKEND` / `DRAFT_KV_DTYPE` | `TRITON_ATTN` / `fp8` | the only fp8-draft pair that boots on GB10 |
| `MAX_NUM_BATCHED_TOKENS` | profile (16384) | chunked-prefill chunk; a multiple of 128 |
| `FAT_THRESHOLD` / `FAT_THRESHOLD_PREFILL` / `FAT_PREFILL_TOKENS` | profile (128 / 0 / 1024) | experts above the row threshold take the grouped fat GEMM; batches of ≥ `FAT_PREFILL_TOKENS` tokens use the prefill threshold. `FAT_THRESHOLD` also caps coop decode eligibility |
| `GPU_MEMORY_UTILIZATION` | `0.85` | startup free-memory gate only; the pool is pinned by `KVB` |
| `GUARD_START_MIB` / `GUARD_MIB` | `2048` / `1024` | MemAvailable floors while loading / once ready |
| `DROP_CACHES_EVERY_S` | `20` | the guard drops page cache this often (GB10 UMA) |
| `API_KEY` | empty | passed to vLLM as `VLLM_API_KEY`, never in argv |
| `ALLOWED_CONTAINERS` | empty | containers `start` tolerates on the nodes; anything else must be stopped first |
| `STOP_OTHER` / `RESTORE_OTHER` | empty | commands to stop and restore another stack that normally owns the GPUs (`./m31.sh stop-other` / `restore-other`) |
| `RANK0_IP` … `FABRIC_IF_1` | empty | the seven link pins (see [Node setup](#node-setup)); set all seven or none |

## Benchmarks and evals

Every tool is stdlib-only and runs from the control machine. [`bench/README.md`](bench/README.md) explains each
measurement.

| Command | What it proves |
|---|---|
| `./m31.sh smoke` | chat coherence, one well-formed tool call, reasoning separated from content, **mixed-K dispatch mode: unified** in rank 0's log |
| `./m31.sh probe` | the surface the bench depends on: models/max_model_len, tokenize, usage, reasoning field, tool parser, spec-decode counters, `reasoning_effort` |
| `./m31.sh bench decode` | concurrency sweep (prose/code/structured/json), temperature 0, fixed windows; aggregate and per-stream tok/s, TTFT; tokens from `usage`, never from chunk counts (DSpark emits up to 4 per delta) |
| `./m31.sh bench prefill` | cold prefills at the given lengths, unique nonce, actual `usage.prompt_tokens` |
| `./m31.sh bench needle` | a 6-character code at chosen depths; PASS only if the code is in content or reasoning |
| `./m31.sh bench spec` | `/metrics` deltas: drafts, accepted draft tokens per verify step (the step emits one more), acceptance rate per prompt class |
| `./m31.sh bench pin` / `check` | teacher-forced NLL of six fixed passages; `check` fails when a passage moves more than 10% |
| `./m31.sh longctx` | salted needles by length × depth, prefill tok/s and TTFT per point |
| `./m31.sh quality` | 50 exact-answer questions (40 seeded multiplications + 10 capitals) |
| `./m31.sh tools` | 27 deterministic tool-calling checks (24 single-call scenarios plus multi-turn follow-ups) |
| `./m31.sh api-test` | 15 OpenAI-surface checks: streaming + usage, max_tokens, stop, reasoning, tools, completions, tokenize, errors, concurrency, determinism, API key |

## The memory guard

`start` arms a root guard on each node before the rank launches.
- It samples `MemAvailable` every second.
- The floor is 2 GiB while loading and 1 GiB once `/v1/models` answers.
- It kills the container after three consecutive breaches, or once swap grows past 64 MiB.
- It drops page cache every 20 s.

On GB10 a host that runs out of memory can take the whole node down, not just a request, so the guard is
mandatory. If `start` dies with `GUARD KILL`, read `~/m31/run/guard.log` and the saved rank log
(`~/m31/run/last.log`). Lower `KVB` or `MAX_MODEL_LEN`; do not lower the floor.

## Known limits

- **Tested hardware:** two pairs of DGX Sparks on one RoCE setup. Other fabrics and driver versions are outside the
  tested envelope.
- **The base image is a dated vLLM dev build** (`0.1.dev23341+g8744dd2fb`) published as
  `vllm/vllm-openai:minimax-m31-flash`. It carries no upstream commit label; the digest is the authority.
- **Boot time:** about 100-110 s to ready from a weight snapshot. The first boot of a new image loads the checkpoint
  and writes the snapshot, taking about 240 s.
- **The NLL check uses short passages** (32-129 tokens). They never reach the long-prefill kernel paths (fat-tier
  GEMM, tiled sparse attention, glue fusion); those rely on unit tests against an fp32 reference and the 16K needle
  test.
- **Adaptive thinking** means output length, and therefore tok/s, depends on the prompt. Every bench pins
  temperature 0 and sends no thinking flag.

## Credits

- [MiniMax](https://www.minimax.io): MiniMax-M3.1-Flash, the base model and the DSpark draft.
- [Mia's AI Lab](https://github.com/MiaAI-Lab) (with @plotarmordev): the fat-GEMM CUDA kernels and routed-expert EXL3
  serving path inside `vllm-exl3` (MIT), which several of this recipe's prefill patches extend.
- [vcruz305](https://huggingface.co/vcruz305):
  - the EXL3 SAGE 2.15 bpw pack;
  - the `vllm-exl3` plugin the patch series applies to;
  - the `exllamav3` fork with the act-3 (clamped SwiGLU-OAI) MoE fix that the exllamav3 patches build on;
  - the one-Spark recipe this repo's structure follows.
- [tonyd2wild](https://github.com/tonyd2wild):
  - the first 2.15 bpw TP=2 lane;
  - plugin patch 0001 (derived from his work) and 0002 (his file, byte for byte);
  - the base flag set of the `q4-300k` profile.
- [turboderp](https://github.com/turboderp-org/exllamav3) (ExLlamaV3), the [vLLM](https://github.com/vllm-project/vllm)
  project, and [Hugging Face](https://huggingface.co).

[`NOTICE`](NOTICE) lists every third-party component and patch with its license.

## Licenses

This repository mixes three licenses:
- **Apache-2.0** ([LICENSE](LICENSE)): the scripts and documentation.
- **AGPL-3.0-only:** the plugin patches (`patches/0001`-`0019`), because `vllm-exl3` is AGPL. If you distribute an
  image built from them, you must also offer the corresponding source.
- **MIT:** the exllamav3 patches.

The engines keep their own licenses: `vllm-exl3` is AGPL-3.0-only and contains MIT-licensed code by Mia's AI Lab
and turboderp, and `exllamav3` is MIT. Their license and notice files (`vllm-exl3`'s `NOTICE` and
`THIRD_PARTY_NOTICES.md`, `exllamav3`'s `LICENSE`) are copied into every image the build makes and must be kept
with any redistribution.

The model weights are not part of this repository. They are under the **MiniMax Community License**
([pack LICENSE](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw/blob/main/LICENSE)), which
grants use for non-commercial purposes and sets conditions on commercial use. Read it before deploying.

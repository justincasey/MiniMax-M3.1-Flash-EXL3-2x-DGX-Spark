# bench — measurement and regression harness

All tools are stdlib-only and run from the control machine against the OpenAI endpoint. None of them write to the
server.

## Layout

| File | Purpose |
|---|---|
| `bench.py` | `decode`, `prefill`, `needle`, `spec`, `pin`, `check`, `all` |
| `quality_gate.py` | 50 exact-answer questions (40 seeded multiplications + 10 capitals) |
| `longctx.py` | salted needle sweep by length and depth; prefill tok/s and TTFT |
| `toolgate.py` | 27 deterministic tool-calling checks (24 single-call scenarios plus multi-turn follow-ups) |
| `probe.py` | capability probe: models/tokenize/chat/reasoning/tools/metrics/effort |
| `runs/` | raw run directories (git-ignored) |
| `records/` | committed baselines and reference records |
| `reference.json` | teacher-forced NLL reference written by `pin` on your stack (`REFERENCE` picks another file) |

## Commands

```bash
./m31.sh probe
./m31.sh bench decode  --conc 1,2,4,8 --types prose,code,structured,json --seconds 30
./m31.sh bench prefill --lengths 8k,32k,128k
./m31.sh bench needle  --tokens 131072 --depths 10,50
./m31.sh bench spec    --spec-types prose,code
./m31.sh bench pin     # writes the reference: mean NLL of six fixed passages, teacher-forced
./m31.sh bench check   # exit 1 when a passage's mean NLL moves more than --nll-tol (10%)
```

Why not an exact token pin: on this stack greedy generation is not reproducible even within one boot. A repeated
temperature-0 request diverged as early as token 0, which points at near-ties in the sparse-attention block
selection. Teacher forcing removes the compounding: per-passage mean NLL moved at most 2.26% across repeats and
0.66% across two Spark pairs, while wrong or corrupt weights move it by far more. The tolerance is 10%.

```bash
./m31.sh quality --out bench/runs/quality.json
./m31.sh longctx --lengths 8k,32k,64k,128k,256k --depths 10,50
./m31.sh tools --out bench/runs/toolgate.json
./m31.sh api-test
```

## Reading the numbers

- **Tokens come from `usage.completion_tokens`**, never from SSE chunk counts. DSpark emits up to k+1 tokens in one
  delta, so chunk counting undercounts decode by roughly the acceptance length.
- **`decode` aggregate** = sum of completion tokens / cell wall time. **Per-stream** = median of
  `(completion_tokens - 1) / (last delta time - first delta time)`.
- **`prefill`** reports the engine's own `prompt_tokens` and wall time; the generated word count is only a sizing
  heuristic. `--word-ratio 1.3` approximates MiniMax's tokenizer.
- **`spec`** divides `/metrics` deltas: accepted tokens / drafts = accepted draft tokens per verify step (the step
  emits one more token from the target, so tokens per step is that plus 1). A value of 0 means the draft is fully
  rejected: a broken DSpark path, not a slow one. Healthy at k=3: about 0.9 on prose and 1.7 on code.
- **Thinking is adaptive.** The client sends no thinking flag; `completion_tokens_details.reasoning_tokens` is
  recorded when exported. Compare runs on the same prompt set, never on tok/s alone.
- **Temperature 0 everywhere.** `pin`/`check` additionally set `seed: 1234` and read `return_token_ids`.

## Records

A record is a committed directory under `records/` containing the run's `results.json` files, the quality and
tool-gate JSON, the profile name, image id, pack revision and the exact `bench.py` command line. `runs/` is
scratch; `records/` is the evidence. Never paste a number into the README without a record behind it.

`PERFORMANCE.md` quotes harness runs and says so; a profile becomes `PROFILE_MEASURED=1` only with a record here.

# First-run runbook

A first hardware pass on your own pair of Sparks, in order. Do not skip a gate.

## Preconditions

```bash
./m31.sh doctor
```

Expected clean state:

- both ranks reachable; 4 KiB pages; docker, rsync, the hf CLI and passwordless sudo present (README, Node setup);
- rank 0 reaches both engine repos on github.com (the build fetches the pinned commits from there);
- the QSFP link detected. On the test pairs this was two subnets, GID 3, and both `rocep1s0f1` +
  `roceP2p1s0f1`;
- no foreign containers. If another stack normally owns the GPUs, put its stop and restore commands in
  `STOP_OTHER` / `RESTORE_OTHER`, run `./m31.sh stop-other`, then `doctor` again;
- ≥ 230 GiB free disk per node, and ≥ 60 GiB `MemAvailable` on rank 0 before the build.

## 1. Fetch (~55 GiB downloaded per node, 110.81 GiB stored per node)

```bash
./m31.sh fetch
```

Artifacts: `~/m31/pack-manifest.tsv`, both nodes' model directories, `~/m31/run/verify.json`. A single bad sha256
fails the fetch. If ssh over the cable is unavailable, the log warns and uses the management LAN.

## 2. Build (rank 0; ships the image over the cable)

```bash
./m31.sh build
```

The build:
- pulls the base image by digest;
- fetches the two pinned engine commits by full sha;
- applies both patch series;
- builds and labels the image with the base, exllamav3, plugin and patch-series shas;
- ships it to rank 1.

Only the two pinned engine commits are fetched (from GitHub, on rank 0); the Docker build itself downloads nothing.
Failures leave the staged context in `~/build/m31`.

## 3. Start (guards → both ranks → ready → warm-up → smoke)

```bash
./m31.sh start
```

The first boot of a new image loads the checkpoint and writes a per-rank weight snapshot, taking about 240 s.
Later boots restore it in about 100 s. Record immediately:

- `runs/<timestamp>/start.txt`: ready `MemAvailable` per rank, profile, KV budget;
- `runs/<timestamp>/kv.txt`: rank 0's KV-capacity line;
- `runs/<timestamp>/warmup.jsonl`;
- the smoke result, including `mixed-K dispatch: unified`.

**Abort criteria:** `GUARD KILL`, a rank exits, `/v1/models` never answers, the smoke test fails, or the engine
reports a KV pool far below the profile's budget. Read `runs/<timestamp>/rank0.log` and `rank1.log` (saved on any
start failure), `~/m31/run/guard.log` and `~/m31/run/last.log`. Lower `KVB` or `MAX_MODEL_LEN`, not the guard
floor.

## 4. Probe and correctness gates

```bash
./m31.sh probe
./m31.sh bench spec --spec-types prose,code --seconds 30
./m31.sh bench needle --tokens 16384 --depths 10,50,90
./m31.sh bench pin      # writes the NLL reference on this stack
./m31.sh bench check    # must stay within 10% of it
./m31.sh quality --out runs/first/quality.json
./m31.sh tools --out runs/first/toolgate.json
./m31.sh api-test
```

Before quoting any speed:
- probe OK;
- spec accepted/step > 0 (accepted draft tokens per verify step; 0 means the draft is rejected);
- needles pass;
- `check` passes;
- quality and toolgate recorded;
- API conformance 0 FAIL.

## 5. Speed and record

```bash
./m31.sh bench decode --conc 1,4 --types prose,code,structured,json --max-tokens 256 --seconds 30 --out runs/first/decode
./m31.sh bench prefill --lengths 8k,16k --out runs/first/prefill
./m31.sh longctx --lengths 8k,32k,64k,128k --depths 10,50 --out runs/first/longctx.json
```

These are the commands behind `PERFORMANCE.md`. To publish a profile as measured, write
`bench/records/<date>-<profile>/` with `meta.json` (image id, pack revision, profile, command lines), the raw
results, and the quality/toolgate JSON. Then set `PROFILE_MEASURED=1` in that profile.

## 6. Context profiles

Pick a `q4-215-*` profile by context and stream count (see the README table). For each one you plan to rely on,
record the ready memory, the guard minimum after a long prefill, the KV capacity line, decode at its stream count,
and needles at its context.

## Rollback

```bash
./m31.sh stop
./m31.sh restore-other   # only if RESTORE_OTHER is set
```

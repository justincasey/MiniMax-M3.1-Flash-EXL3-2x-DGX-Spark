# Agent Instructions — MiniMax-M3.1-Flash EXL3 2x DGX Spark

This is a serving-recipe repository, not an application. A change is correct only when the pinned stack still
serves and the measurement record says so.

## Read first

- `README.md` — the lane, the pins, the profiles, the measured numbers (`PERFORMANCE.md` has the detail).
- `bench/README.md` — what each measurement means and how it is produced.
- `patches/README.md` — both patch series (plugin and exllamav3), why each exists, its license.
- `docs/RECIPE_DECISIONS.md` — decisions and traps from building this lane.
- `docs/FIRST-RUN.md` — the first hardware pass, in order, with abort criteria.

## Rules

- **Pins live in `config/pins.sh`.** Do not tune them from `.env`. A pin change is a release: re-run the contract
  tests, rebuild, re-verify, update `VALIDATED_TREES` in `tests/test_pack_layout.py`, and start a new record. The
  engine sources are fetched at full commit shas; never pin a branch, a tag or an abbreviated sha.
- **No invented numbers.** The README quotes only byte censuses, labelled estimates, or a committed record under
  `bench/records/`. If a number has no record, label it unmeasured or omit it.
- **Fail closed.** `start` refuses missing/oversized files, label mismatches, partial fabric pins, foreign
  containers and unsafe KV combinations. Keep it that way; do not add a flag that skips a gate.
- **The memory guard is mandatory.** On GB10 a host OOM takes the node down. Do not lower the floor to fit a
  profile; lower `KVB` or `MAX_MODEL_LEN`.
- **One layout for the Docker context.** `prepare_src.sh` stages `exllamav3-src/`, `vllm-exl3-src/`, `Dockerfile`
  and `PINS` inside `$BUILD_DIR/docker`; `docker build` uses that directory as the context.
- **Remote scripts receive variables by name.** `remote()` seeds the prelude; it passes no positional arguments.
  Never read `$1` in a script sent through `remote`/`both` (see `tests/test_recipe.sh`).
- **The serve config is merged, never replaced.** `scripts/make_serve_config.py` keeps the pack's model definition
  and swaps only `quantization_config`. Mounting the overlay alone erases the model.
- **Adaptive thinking.** Do not add a thinking flag to every request. Pin temperature 0 in benchmarks; record
  `reasoning_tokens` when exported.

## Before finishing

```bash
bash tests/test_recipe.sh
python3 tests/test_pack_layout.py          # also fetches both pinned commits and applies both patch series
python3 -m py_compile bench/*.py scripts/*.py tests/*.py
```

After a hardware run, update the record, not the prose: put `results.json` and `meta.json` under
`bench/records/<date>-<profile>/` and only then quote a number in the README.

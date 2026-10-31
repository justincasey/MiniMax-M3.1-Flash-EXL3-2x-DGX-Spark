# Records

Committed measurement evidence. Each record directory is one boot configuration:

```
records/<date>-<profile>/
  meta.json          profile, image id, pack revision, GPU clocks, command lines
  decode/            results.json + summary.md
  prefill/           results.json + summary.md
  needle/            results.json + summary.md
  spec/              results.json + summary.md
  check/             results.json + summary.md
  quality.json
  toolgate.json
  longctx.json
```

`meta.json` is the provenance: without it a record is not evidence. The repository ships **no** records yet (the
numbers in `PERFORMANCE.md` are labelled harness runs); `bench/runs/` is git-ignored scratch. A profile becomes
`PROFILE_MEASURED=1` only once its record is committed here.

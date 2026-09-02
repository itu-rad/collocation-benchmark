# Evaluation

Experimental setup, collection harnesses and analysis for the paper's experiments.

**This file stays high-level on purpose.** Each experiment owns a write-up next to its code,
and that is where the detail, the commands and the numbers live — so this page does not need
updating every time a result moves.

| § | experiment | directory | write-up |
|---|---|---|---|
| §4 | E1 — framework overhead (NoOp) | `overheads/framework_overhead/` | [`framework_overhead.md`](overheads/framework_overhead/framework_overhead.md) |
| §4 | E2 — modularity overhead (EfficientNetV2) | `overheads/modularity_overhead/` | [`modularity_overhead.md`](overheads/modularity_overhead/modularity_overhead.md) |
| §4 | E3 — MLPerf 3D-UNet / KiTS19 | `unet3d/` | [`unet3d.md`](unet3d/unet3d.md) |
| §5.1 | Self-RAG execution strategies | `self_rag/` | [`self_rag.md`](self_rag/self_rag.md) |
| §5.2 | Collocation types | `contention/` | [`contention.md`](contention/contention.md) |

Supporting: `pilots/` (pre-registered knob derivation and warm-up convergence),
`scripts/` (quality scoring and cross-cutting analysis), `radt-patches/` (patches that must be
applied to the installed radt).

The authoritative statement of what each experiment claims is `../EXPERIMENTS.md`.

## Conventions

Every experiment follows the same shape, so a reader who learns one can read the rest:

- **One collection harness** — `collect_<experiment>.sh`. Provenance header (git commit and
  dirty flag, host, platform, library versions, pinning, run counts), timestamped logs that are
  never appended to, `caffeinate` on macOS, stale-file clearing, and an occupancy gate that
  refuses to start on a busy machine.
- **One analyzer** — `analyze_<experiment>.py`. Self-contained: parsing, statistics, tables,
  LaTeX and figures in a single file, so the write-up and the analyzer cannot disagree. It
  must reproduce every table and figure from a bare invocation.
- **Results live beside the experiment** that produced them, in its own `results/`. Each
  harness exports `BENCH_OUTPUT_DIR`, so the run's CSV and JSONL are written straight there —
  there is no shared staging directory and nothing to sweep afterwards. Running `main.py` by
  hand without that variable falls back to `evaluation/results/`.
- **Figures** go in each experiment's `paper_assets/`.
- **Statistics.** The run is the unit of replication; confidence intervals are a hierarchical
  bootstrap that resamples runs first, then queries within a run. Repetition 1 is dropped as
  system warm-up — the first repetition is slower for its whole duration.
- **Tracking store.** Everything records to res17 (experiment 138), except E1 and E2, which use
  a local MLflow store: the overhead microbenchmarks emit spans at a rate no real workload
  approaches, and measuring that against a remote server measures the server.

## Trace format

One event per line, comma-separated, no header:

```
<wall_timestamp>, <pipeline_name>, <stage_name>, <phase>, <state>[, <extras...>], <perf_counter_ns>
```

`phase` ∈ {`prepare`, `run`}, `state` ∈ {`start`, `end`}. The trailing monotonic
`perf_counter_ns` is what timing uses; column 0 is wall clock, kept for cross-process and
listener alignment only. Spans carry the same monotonic value as a `perf_start_ns` attribute.

Stages with `disable_logs` emit no per-query rows — timing then comes from spans alone.

# Evaluation

This directory contains the experimental setup, runners, and analysis tooling
for the studies the Choreo paper relies on (`../EXPERIMENTS.md` is the
authoritative list and numbering; the sections below are ordered by
directory, not by experiment number). Each experiment produces
plain-text CSV traces (and, for the complex cases, a JSONL sidecar with
per-query outputs); analysis scripts read those traces and emit Markdown or
LaTeX reports.

## Layout

```
evaluation/
├── overheads/        # instrument-fidelity studies that bound the
│   │                 # framework's own overhead
│   ├── framework_overhead/   # noop sweep — self-contained: generator +
│   │                         # analyzers + configs/ + results/
│   └── modularity_overhead/  # standalone PyTorch that re-implements a
│                             # Choreo workload for an honest A/B
├── unet3d/           # E3 — MLPerf 3D-UNet / KiTS19: parity with MLPerf's
│                     # own reference harness, then the share its offline
│                     # measurement boundary hides. Self-contained:
│                     # collect_e3.sh + analyze_e3.py + configs/ + results/
├── self_rag/         # §5.1 — Self-RAG execution strategies (case study)
├── contention/       # §5.2 — collocation types, per-pipeline attribution
├── pilots/           # pre-registered knob derivation + warm-up convergence
├── scripts/          # analysis scripts for the case studies (no execution
│                     # side effects beyond writing files into results/)
└── results/          # all CSV/JSONL traces from runs + all Markdown/LaTeX
                      # reports from analyzers. Gitignored — reproducible.
```

The pipeline YAMLs for the complex cases live one level up in
`../pipeline_configs/` to be consistent with the rest of the codebase.

## Experiments

### 1. Framework overhead — NoOp microbenchmark

**What:** chains of N pass-through stages bound the framework's own measurement
overhead. It separates two costs: the **core dispatch** (thread wake + queue
hand-off + CSV log, measured with tracing disabled) and the **MLflow tracing
layer** (3 spans/stage, measured with tracing on). Three results: depth-flatness,
zero-copy (reference O(1) vs deep-copy O(payload)), and overhead-in-context.
Everything lives under `overheads/framework_overhead/`; the write-up is
`framework_overhead.md`.

**Collect** (`collect_e1.sh` runs the whole depth sweep, both configurations, R
repetitions):
```bash
# on the M3 Pro, inside the project env
bash evaluation/overheads/framework_overhead/collect_e1.sh m3pro 11 143
# on the GB10 — pin to one X925 performance core, or the Grace scheduler
# migrates the run between core types and the timing goes bimodal
bash evaluation/overheads/framework_overhead/collect_e1.sh gb10 11 143
```
Two configurations, named for the role each plays rather than the switch that
sets it — they are the only two ways the framework is actually run:

| configuration | CSV stage logging | spans | what it is |
|---|---|---|---|
| `uninstrumented` | off | off | the framework bare |
| `+ tracing` | off | on | the framework as it is actually run |

`+ tracing − uninstrumented` is therefore what tracing costs. Per-query latency
`L_q` survives with stage logging off because it comes from the pipeline-level
rows, which `pipeline.py` emits unconditionally. CSVs land in
`evaluation/results/<machine>/noop_depth_D_size_S_mode_M_<configuration>_<machine>_rN.csv`;
a timestamped log and summary go to `collect_logs/`, headed by a provenance
block (git commit + dirty flag, host, platform, python/torch/radt/mlflow, run
and step counts). `E1_PAYLOAD=1` collects the payload sweep instead (depth 10 ×
{0, 1 KiB, 1 MiB, 10 MiB} × {ref, copy}).

**Analyze** (one self-contained analyzer — tables, statistics and figures):
```bash
python evaluation/overheads/framework_overhead/analyze_e1.py
python evaluation/overheads/framework_overhead/analyze_e1.py --latex m3pro > tables.tex
```
Timing uses the monotonic `perf_counter_ns` trailing CSV column (wall-clock
column 0 is kept only for radt alignment); CIs are a hierarchical bootstrap with
the RUN as the unit of replication. Figures go to `framework_overhead/paper_assets/`.
To run one cell by hand: `python main.py <config> -p 0 --label <label>`.

Overhead experiments (E1 and E2 only) record to a LOCAL MLflow store rather than
res17 — the microbenchmark emits far more spans per second than any real
workload, and that volume is not what the remote server is there to carry.

### 2. Modularity overhead — EfficientNetV2 monolith vs Choreo

**What:** the same EfficientNetV2 Imagenette fine-tune expressed two ways — a
hand-written PyTorch monolith vs the Choreo pipeline — to measure what the
framework's graph/queue/thread wrapper costs on a real workload. The
real-workload counterpart to the NoOp experiment. Everything lives under
`overheads/modularity_overhead/`; the write-up is `modularity_overhead.md`.

Two things run per cell, on an identical workload (the `(model, weights,
batch)` triple is read from the same YAML both sides are given):

| configuration | what it is |
|---|---|
| `monolith` | `baseline_finetune.py` — no framework at all, no tracing |
| `choreo-traced` | `main.py` with `CHOREO_PROC_TRACE=1` — the framework as it is actually run |

`choreo-traced − monolith` is the cost of decomposition, tracing included.
Their order is rotated by repetition index so neither always absorbs the
warm-up. There is deliberately no third, untraced Choreo configuration: tracing
is how the framework is run, splitting the cost in two doubled the collection
for a term that was within noise at every cell, and the split is measured
directly and far more precisely by E1.

**Metric of record: time per query** — start-to-start between consecutive
queries, i.e. 1/throughput, covering the whole cycle including data loading and
preprocessing. It is anchor-invariant in steady state (the same period comes out
of the pipeline row, the training row or the monolith's step row), which is what
makes the two processes comparable although they emit different markers. The
earlier metric — the training step's own duration compared across processes — is
not usable: the framework's cost lands mostly *between* steps, which that marker
excludes by construction, so it measured a near-zero difference against ±600 µs
of run-to-run noise and flipped sign between repetitions.

**Co-headline: the query latency breakdown**, from the spans of the
`choreo-traced` runs — per-stage latency (dataloader, training) plus the
auxiliary framework overheads (entry, handoff, exit, turnaround). Those are
successive instants within one query on one clock, so they are non-negative by
construction and carry no run-level term; they sum to the time per query exactly
(E1 verified the identity at a residual of 0.000 µs over 300 queries).

**Choreo writes no per-query CSV rows at all.** Both flags are set on the E2
configs — `disable_logs` on the pipeline (the per-query rows) and on each stage
(the per-stage rows) — so every Choreo number in E2 comes from spans. This is
not cosmetic: the synchronous write+flush was 42–50% of the measured `exit`
term before it was gated off. The monolith keeps its own log because it is the
only instrument a bare loop has; the asymmetry is the point of the control, and
it means the two sides are compared on `in-pipeline L_q` (spans, admission
excluded) against `time per query` (the monolith's own steps).

**Collect (one harness, both configurations, R repetitions; per machine):**
```bash
python evaluation/overheads/modularity_overhead/gen_configs.py   # regenerate cells
bash evaluation/overheads/modularity_overhead/collect_e2.sh m3pro 11
# on the GB10, pin as in E1
PIN=19 bash evaluation/overheads/modularity_overhead/collect_e2.sh gb10 11
```
Sweep: EfficientNetV2-S at batch {1,2,4,8,16,32,64}, plus EfficientNetV2-M and -L
at batch 8 — 9 cells, 300 steps per run. CSVs land in
`results/mod_<cell>_<configuration>_<machine>_rN.csv`, with a timestamped log,
summary TSV and provenance header in `collect_logs/`. Local MLflow store, as in §1.

**Analyze:**
```bash
python evaluation/overheads/modularity_overhead/analyze_e2.py
python evaluation/overheads/modularity_overhead/analyze_e2.py --latex gb10 > table2.tex
```
Warmup 50 queries per run; **run 1 of every cell is dropped by default** — the
first repetition is measurably slower for its whole duration, so per-step warmup
cannot remove it, and collection runs R+1 to leave R usable.

Two statistics are reported, and they answer the same question with very
different resolution:

- **Cost of decomposition** — the paired across-run difference between the
  monolith's time per query and Choreo's in-pipeline `L_q` (runs pair by id
  because the configurations are interleaved within each repetition), combined
  by median, bootstrapped over run PAIRS. It is the honest end-to-end
  comparison, and at most cells its interval contains zero: the cost is below
  what a cross-process comparison resolves.
- **Framework term** — `entry + handoff + exit`, measured from spans on one
  clock inside the Choreo process. Same bootstrap, but no cross-process drift,
  so the interval is orders of magnitude tighter. This is what the figure plots.

Figures go to `modularity_overhead/paper_assets/`.

### 3. MLPerf 3D-UNet / KiTS19 — parity, then the measurement boundary

**What:** two prongs. Reproduce MLPerf's own reference harness with Choreo on the
same machine (GB10) on accuracy and performance, then show what its offline
measurement boundary hides online — MLPerf preloads preprocessed volumes and
times only inference, but an arriving request has nothing to prefetch. Everything
lives under `unet3d/`; the write-up is
[`unet3d.md`](unet3d/unet3d.md), which carries the full method, the reasons for
the two-config split, and the objections it has to keep answered.

**Timing is spans only** and **E3 runs UNPINNED**, unlike E1 and E2 — pinning
throttles CPU preprocessing but not GPU inference, which would inflate the very
ratio the experiment reports, so `collect_e3.sh` refuses a `PIN`.

```bash
bash evaluation/unet3d/collect_e3.sh gb10  1 acc   # accuracy pass, never timed
bash evaluation/unet3d/collect_e3.sh gb10  6       # timed, repetition 1 dropped
bash evaluation/unet3d/collect_e3.sh m3pro 1 acc
bash evaluation/unet3d/collect_e3.sh m3pro 6
python evaluation/unet3d/analyze_e3.py --machines m3pro gb10
```

Records to res17 (experiment 138), not a local store — the local-store exemption
covers the overhead experiments only.

### 4. Self-RAG — execution strategies (§5.1 case study)

**What:** the same agentic Self-RAG job (grade → answer → hallucination-check, with a
retry back-edge) executed four ways, differing **only in YAML**: a monolithic prompt; one
model shared behind a lock (`depends_on_id`); per-role copies; and one model behind a
server with continuous batching. Two tasks (factoid, multi-hop) × two devices.

Register is **investigative, not consumer** — the deliverable is a causal account of where
time and memory go under each strategy, not a verdict on which is fastest. See
[`self_rag/README.md`](self_rag/README.md) for the arms, exhibits and data protocol, and
`../EXPERIMENTS.md` (E4 / §5.1) for the authoritative framing.

**Note:** the run commands that used to live here were stale. The collection harness is
being reworked to the `collect_e3.sh` pattern (`collect_e4.sh`); until it lands, see
`self_rag/collect.sh`.

### 5. Collocation types (§5.2 case study)

**What:** the §5.1 Self-RAG serving pipeline as foreground, plus a background pipeline,
sweeping the **collocation type** — same engine / ANE / CPU on m3pro; time-sliced GPU /
MPS / CPU on gb10. The background workload is a prop; the subject is what the foreground
actually contends on, and whether interference can be **attributed per pipeline** (each
pipeline is its own process with its own radt run and listeners). Code in
[`contention/`](contention/); hardware facts in `../CONTENTION_EXPERIMENTS_REDESIGN.md`.

## How runs work in general

Every pipeline run goes through `main.py`, which:

1. Parses the YAML into a `BenchmarkModel`.
2. Applies CLI overrides:
   - `-p <id>` — pipeline index to run (default `-1` hands off to RadT).
   - `--serialize {true,false}` — force-override `serialize_queries` on
     the selected pipeline (no need to duplicate the YAML).
   - `--label <suffix>` — override the per-run output filename so multiple
     runs of the same config land in distinct CSV/JSONL files (propagated
     to `TerminalCapture` via `CHOREO_OUTPUT_LABEL` env var).
3. Sets up the per-run CSV log handler under `evaluation/results/`.
4. Hands the parsed config to `loadgen.run_loadgen(...)`, which spawns
   stage threads, drives them via the scheduler, and joins on completion.
5. Force-exits with `os._exit(0)` after the pipeline finishes — needed
   because mlflow telemetry sockets, joblib/loky semaphores, and MLX
   Metal teardown otherwise hold the process alive for tens of minutes
   despite all results already being on disk.

## How analyzers work

All analyzers are pure file-based: they read CSV / JSONL files from
`evaluation/results/` and write Markdown / LaTeX reports back into the
same directory. No execution side effects beyond the report file.

| Script | Reads | Writes |
|---|---|---|
| `verify_complex_cases.py` | `<pipeline>_outputs.jsonl` (Self-RAG) | `verification_report.md` |
| `score_quality.py` | `<label>_outputs.jsonl` | `quality_report.{md,json}` — EM/F1, the §5.1 quality column |

The overhead experiments are self-contained under `overheads/`, each with its
own analyzers + `results/`:

Each is one collection harness plus one self-contained analyzer — parsing,
statistics, tables, LaTeX and figures in a single file, so the `.md` and the
`.tex` cannot disagree:

- **framework_overhead** — `collect_e1.sh` + `analyze_e1.py`, reading
  `evaluation/results/<machine>/noop_depth_*_<configuration>_<machine>_r*.csv`.
  Configs are generated by `noop_chain_generator.py` and `gen_nolog_configs.py`.
- **modularity_overhead** — `collect_e2.sh` + `analyze_e2.py`, reading
  `results/mod_<cell>_<configuration>_<machine>_r*.csv`. Configs are generated
  by `gen_configs.py`; the monolith control is `baseline_finetune.py`.

`unet3d/` follows the same shape — `collect_e3.sh` + `analyze_e3.py` — but reads
SPANS from res17 rather than CSVs from disk, because its configs write nothing
per query.

Both take `--machines m3pro gb10` (the MACHINE, not the torch device string —
that lives inside the config) and read the monotonic perf column.

## Output format reference

### Timing CSV (every pipeline run)

One event per line, comma-separated, no header:

```
<timestamp>, <pipeline_name>, <stage_name>, <phase>, <state>[, <extras...>]
```

- `phase` ∈ {`prepare`, `run`}
- `state` ∈ {`start`, `end`}
- For per-query events, the stage_name is `pipeline - <split>` and the
  extras are: `<query_id>, <submitted_ts>, <epoch>, <batch_idx_one_based>`.

Per-query latency = end-timestamp − start-timestamp for matching
`pipeline - <split>, run, start/end` pairs.

### JSONL sidecar (complex-case pipelines only)

Written by `stages.TerminalCapture` at the end of each pipeline. One
object per completed query:

```json
{
  "query_id": "...",
  "epoch": 1,
  "batch": 3,
  "split": "val",
  "question": "...",
  "golden_answers": ["..."],
  "retrieved_documents": ["...", "...", "..."],
  "generated_answer": "...",
  "final_data": "..."
}
```

`generated_answer` is what the answer-producing stage put in
`query.context["generated_answer"]`. `final_data` is whatever sat in
`query.data` when the query reached the end stage (typically the same as
`generated_answer` for accept-paths, or an error marker for
retry-exhausted paths).

## Reproducing a run from scratch

```bash
# 1. environment (Apple Silicon)
conda env create -f environments/macos.yaml
conda activate benchmark_macos

# 2. pick the experiment of interest and follow its section above
```

First runs of any pipeline will download HF datasets and model weights
into `~/.cache/huggingface/`. Budget ~10 GB for Qwen 3.5-9B-OptiQ +
3 GB for Qwen 3.5-4B-OptiQ + ~1.5 GB for CLIP-ViT-L/14 + small dataset
metadata.

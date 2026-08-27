# Evaluation

This directory contains the experimental setup, runners, and analysis tooling
for the four studies the Choreo paper relies on. Each experiment produces
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
# on the M2 Pro, inside the project env
bash evaluation/overheads/framework_overhead/collect_e1.sh m2pro 11 143
# on the GB10 — pin to one X925 performance core, or the Grace scheduler
# migrates the run between core types and the timing goes bimodal
bash evaluation/overheads/framework_overhead/collect_e1.sh gb10 11 143
```
Two configurations, named for the role each plays rather than the switch that
sets it — they are the only two ways the framework is actually run:

| configuration | CSV stage logging | spans | what it is |
|---|---|---|---|
| `uninstrumented` | off | off | the framework bare |
| `spans-only` | off | on | the framework traced |

`spans-only − uninstrumented` is therefore what tracing costs. Per-query latency
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
python evaluation/overheads/framework_overhead/analyze_e1.py --latex m2pro > tables.tex
```
Timing uses the monotonic `perf_counter_ns` trailing CSV column (wall-clock
column 0 is kept only for radt alignment); CIs are a hierarchical bootstrap with
the RUN as the unit of replication. Figures go to `framework_overhead/paper_assets/`.
To run one cell by hand: `python main.py <config> -p 0 --label <label>`.

Overhead experiments (E1 and E2 only) record to a LOCAL MLflow store rather than
res17 — the microbenchmark emits far more spans per second than any real
workload, and that volume is not what the remote server is there to carry.

### 2. Modularity overhead — EfficientNetV2-S monolith vs Choreo

**What:** the same EfficientNetV2 Imagenette fine-tune expressed two ways — a
hand-written PyTorch monolith vs the Choreo pipeline — to measure what the
framework's graph/queue/thread wrapper costs on a real workload. The
real-workload counterpart to the NoOp experiment. Everything lives under
`overheads/modularity_overhead/`; the write-up is `modularity_overhead.md`.

Three things run per cell, on an identical workload (the `(model, weights,
batch)` triple is read from the same YAML both sides are given):

| configuration | what it is |
|---|---|
| `monolith` | `baseline_finetune.py` — no framework at all |
| `choreo` | `main.py` with `CHOREO_DISABLE_TRACING=1` |
| `choreo-traced` | `main.py` with `CHOREO_PROC_TRACE=1` |

`choreo − monolith` is the cost of decomposition; `choreo-traced − choreo` is
the cost of turning tracing on. Their order is rotated by repetition index so
none of them always absorbs the warm-up.

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

Both Choreo configurations run with `disable_logs: true` so that no synchronous
write+flush sits inside the measured interval on one side only; the monolith
writes through the same `FileHandler` `main.py` installs, so the instrument
matches on both sides.

**Collect (one harness, all three configurations, R repetitions; per machine):**
```bash
python evaluation/overheads/modularity_overhead/gen_configs.py   # regenerate cells
bash evaluation/overheads/modularity_overhead/collect_e2.sh m2pro 11
bash evaluation/overheads/modularity_overhead/collect_e2.sh gb10  11
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
cannot remove it, and collection runs R+1 to leave R usable. Statistic of
record: the paired across-run difference (runs pair by id because the
configurations are interleaved within each repetition), combined by median, with
a bootstrap that resamples run PAIRS and re-resamples queries within each.
Figures go to `modularity_overhead/paper_assets/`.

### 3. Multimodal VQA — unified-memory bandwidth contention (Apple Silicon)

**What:** image-grounded VQA pipeline (CLIP-Large vision encode →
FAISS over 10 k COCO captions → Qwen 3.5-9B answer) run under two
accelerator mappings and two scheduling regimes, yielding a 2×2 cell
that isolates the bandwidth-contention effect.

| Cell | Mapping | Schedule |
|---|---|---|
| `vqa_a_pipe` | A: CLIP on MPS, LLM on MPS | pipelined (multiple queries in flight) |
| `vqa_a_serial` | A | `--serialize true` (one query end-to-end at a time) |
| `vqa_b_pipe` | B: CLIP on ANE via CoreML, LLM on MPS | pipelined |
| `vqa_b_serial` | B | `--serialize true` |

The heterogeneity advantage (B vs A) under contention vs without it
exposes whether the unified-memory bandwidth is the binding constraint.

**Prerequisites:**

```bash
# CoreML vision tower export (one-shot, ~3 min)
python stages/multimodal_vqa/export_clip_coreml.py \
  --model openai/clip-vit-large-patch14 \
  --output tmp/clip_vit_l14_vision.mlpackage
```

**Run the 2×2:**
```bash
for cfg in multimodal_vqa_mapping_a multimodal_vqa_mapping_b; do
  for sched in false true; do
    label="vqa_${cfg##*_}_$( [ "$sched" = "true" ] && echo serial || echo pipe )"
    python main.py pipeline_configs/${cfg}.yml -p 0 \
      --label "$label" --serialize "$sched"
  done
done
```

Each run produces `<label>.csv` (timing) and `<label>_outputs.jsonl`
(per-query answer capture from `TerminalCapture`).

**Analyze:**
```bash
# Semantic verification: did the pipeline produce sensible answers?
python evaluation/scripts/verify_complex_cases.py

# Bandwidth analysis: 2×2 latency / throughput / device-busy report
python evaluation/scripts/bandwidth_analysis.py --cells
```

Reports land at `results/verification_report.md` and
`results/bandwidth_report.md`.

### 4. Self-RAG — topology comparison (monolith vs decomposed)

**What:** two Self-RAG pipelines that do the same job with different
decompositions:

- **Monolith:** one large model (9B) does grade + answer +
  hallucination-check in a single JSON pass. `MonolithRouter` validates
  the JSON and optionally loops through a query rewriter.
- **Decomposed:** three distinct 4B instances split the same job into
  separate stages (grader / generator / hallucination-grader, with the
  rewriter sharing the grader), overlapping under load.

The comparison is whether decomposition gives or costs anything, on both
an easy (factoid) and a hard (multi-hop HotpotQA) task. Configs, run
commands, and the results report all live in
[`self_rag/`](self_rag/README.md).

**Run:** see [`self_rag/README.md`](self_rag/README.md) for the full
per-experiment commands. In brief, from the repo root:
```bash
python main.py evaluation/self_rag/configs/factoid_monolith_cuda.yml   -p 0 --label self_rag_monolith
python main.py evaluation/self_rag/configs/factoid_decomposed_cuda.yml -p 0 --label self_rag_decomposed
python evaluation/scripts/verify_complex_cases.py
```

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
| `verify_complex_cases.py` | `<pipeline>_outputs.jsonl` (VQA, Self-RAG) | `verification_report.md` |
| `bandwidth_analysis.py` | timing CSVs (default `vqa_a/b_pipe/serial`) | `bandwidth_report.md` |

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

Both take `--machines m2pro gb10` (the MACHINE, not the torch device string —
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

# 2. one-off CoreML export (for VQA mapping B)
python stages/multimodal_vqa/export_clip_coreml.py \
  --model openai/clip-vit-large-patch14 \
  --output tmp/clip_vit_l14_vision.mlpackage

# 3. pick the experiment of interest and follow its section above
```

First runs of any pipeline will download HF datasets and model weights
into `~/.cache/huggingface/`. Budget ~10 GB for Qwen 3.5-9B-OptiQ +
3 GB for Qwen 3.5-4B-OptiQ + ~1.5 GB for CLIP-ViT-L/14 + small dataset
metadata.

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

**Collect (one driver does the whole matrix, both arms, R runs):**
```bash
# on the M2 Pro, inside the project env
python evaluation/overheads/framework_overhead/run_matrix.py --runs 5
```
`run_matrix.py` generates any missing configs, runs the depth sweep
`{1..10,16,32,50,64,100}` × {tracing off, on} and the payload sweep (depth 10 ×
{0,1KiB,1MiB,10MiB} × {ref,copy}, tracing off), sets `CHOREO_DISABLE_TRACING` for
the off arm, curates each CSV into `results/noop_d{D}_s{S}_m{M}_t{0|1}_r{R}.csv`,
and writes `run_matrix_env.txt` (perf_counter resolution + CPU/mem SKU). It is
resumable (skips existing CSVs unless `--force`).

**Analyze** (all share `noop_lib.py` so the `.md` and `.tex` never disagree):
```bash
python evaluation/overheads/framework_overhead/analyze_noop_results.py --arm both
python evaluation/overheads/framework_overhead/analyze_payload_results.py --fig
python evaluation/overheads/framework_overhead/generate_latex_results.py > tables.tex
```
Timing uses the monotonic `perf_counter_ns` trailing CSV column (wall-clock
column 0 is kept only for RadT alignment). To run one cell by hand:
`python main.py <config> -p 0 --label <label>` (prefix `CHOREO_DISABLE_TRACING=1`
for the core-dispatch arm).

### 2. Modularity overhead — EfficientNetV2-S monolith vs Choreo

**What:** the same EfficientNetV2-S Imagenette fine-tune expressed two ways — a
hand-written PyTorch monolith vs the Choreo pipeline — to show the framework's
graph/queue/thread wrapper adds per-step overhead *within noise*. The
real-workload counterpart to the NoOp experiment. Everything lives under
`overheads/modularity_overhead/`; the write-up is `modularity_overhead.md`.
Same two-arm design (core dispatch / tracing) and monotonic perf clock as §1.

**Collect (one driver, both arms, R runs; run on each DUT):**
```bash
# inside the project torch+cuda env (e.g. benchmark_engines)
python evaluation/overheads/modularity_overhead/run_modularity.py --device cuda --runs 5
python evaluation/overheads/modularity_overhead/run_modularity.py --device mps  --runs 5   # M2 Pro
```
Runs the baseline (R times, `--no-radt` = zero-framework control) and the Choreo
pipeline (R times × {tracing off, on}), patches the config's `device`/`max_queries`
per run, curates CSVs into `results/mod_{baseline,choreo_t{0,1}}_d{dev}_r{R}.csv`,
and writes `run_modularity_env.txt` (perf resolution + torch/cuda/SKU). Resumable.

**Analyze** (all share `modularity_lib.py`; pass `--device cuda|mps`):
```bash
python evaluation/overheads/modularity_overhead/analyze_operational_overhead.py --device cuda
python evaluation/overheads/modularity_overhead/breakdown_overhead.py           --device cuda
python evaluation/overheads/modularity_overhead/true_overhead_analysis.py        --device cuda
python evaluation/overheads/modularity_overhead/generate_latex_results.py        --device cuda > table2.tex
```
Per-step metric: baseline `training_step` vs Choreo `EfficientNet training` (same GPU
work, data loading excluded), monotonic perf column, pooled R runs, warmup 100,
two-sample bootstrap CI; "within noise" = ratio CI contains 0.

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

- **framework_overhead** — `run_matrix.py` (driver) + `noop_lib.py`; analyzers
  `analyze_noop_results.py`, `analyze_payload_results.py`, `generate_latex_results.py`
  read `results/noop_d*_s*_m*_t*_r*.csv`.
- **modularity_overhead** — `run_modularity.py` (driver) + `modularity_lib.py`; analyzers
  `analyze_operational_overhead.py` (headline within-noise), `breakdown_overhead.py`
  (dataloader/end-to-end), `true_overhead_analysis.py` (overhead-in-context),
  `generate_latex_results.py` (Table 2) read `results/mod_*_d{cuda,mps}_r*.csv` via the
  monotonic perf column. Take `--device cuda|mps`.

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

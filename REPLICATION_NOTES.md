# Replication notes — `evaluation/self_rag` on Apple Silicon (MLX)

This document is a step-by-step, reproduce-from-scratch log for the
**Self-RAG monolith-vs-decomposed** case study
(`evaluation/self_rag/`) on an **Apple Silicon** Mac using the **MLX**
backend. It records every command that worked, every hurdle hit, and the
fix, so a fresh clone runs the evaluation without surprises.

Companion docs (authored by the upstream authors):
- `evaluation/self_rag/README.md` — experiment index + the canonical command list.
- `evaluation/self_rag/monolith_vs_decomposed_m2pro.md` — the Apple-Silicon results this run reproduces.
- `evaluation/self_rag/monolith_vs_decomposed_dgxspark.md` — the CUDA/DGX results.

---

## TL;DR (the happy path that works)

```bash
# 1. Clone WITH submodules (the macOS env references stages/anemll/Anemll)
git clone --recurse-submodules https://github.com/itu-rad/collocation-benchmark/
cd collocation-benchmark

# 2. Create the conda env from the PINNED macOS file (see "Hurdle 2")
conda env create -f environments/macos.yaml
conda activate benchmark_macos

# 3. Run the two factoid pipelines (models auto-download from HuggingFace
#    on first run: ~6.6 GB for the 9B, ~3.1 GB for the 4B).
#    NOTE the `--serialize true` on the decomposed run — on a 16 GB Mac the
#    default (pipelined) decomposed run OOMs the GPU. See "Hurdle 7".
python main.py evaluation/self_rag/configs/factoid_monolith_mlx.yml   -p 0 --label self_rag_monolith
python main.py evaluation/self_rag/configs/factoid_decomposed_mlx.yml -p 0 --serialize true --label self_rag_decomposed

# 4. Score answer quality (writes evaluation/results/verification_report.md)
python evaluation/scripts/verify_complex_cases.py self_rag_monolith self_rag_decomposed
```

Outputs land in `evaluation/results/` (gitignored):
`<label>.csv` (per-stage timing trace) and `<label>_outputs.jsonl`
(per-query question / retrieved docs / answer).

---

## Environment that this was verified on

- **Machine:** Apple Silicon (`arm64`), macOS (Darwin 25.x), 16 GB unified memory.
- **conda:** miniconda3, `conda env create` (classic solver), channel `defaults`.
- **Resulting env** `benchmark_macos`, Python 3.10.20, key pins resolved:
  torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0, transformers 5.2.0,
  datasets 4.6.0, chromadb 1.5.9, mlflow 3.10.0, radt 0.2.28,
  **mlx 0.31.2 / mlx-lm 0.31.3**. `torch.backends.mps.is_available() == True`.

---

## Step-by-step, with hurdles

### Step 0 — Pick the right path (Apple Silicon → MLX)

The repo supports two backends. On Apple Silicon you want the **MLX**
configs (`*_mlx.yml`), not the `*_cuda.yml` ones. `uname -m` → `arm64`
confirms you're on the MLX path.

### Hurdle 1 — README filename / env-name drift

- The top-level `README.md` says `conda env create -f environments/macos.yml`,
  but the file is **`environments/macos.yaml`** (`.yaml`, not `.yml`).
- The same README says `conda activate benchmark_macos`, while
  `evaluation/self_rag/README.md` says `conda activate benchmark`. The
  **actual env name** (from the yaml's `name:` field) is **`benchmark_macos`**.

Use `environments/macos.yaml` and `conda activate benchmark_macos`.

### Hurdle 2 — the macOS env file was un-pinned and resolved to broken versions

This is the big one, and it mirrors what the authors already fixed for
the CUDA env. `environments/nvidia.yaml` carries a header explaining that
the previously un-pinned `torchaudio` / `transformers` "drifted to
incompatible versions on fresh installs and broke all model loading."

The original `environments/macos.yaml` had the **same un-pinned** entries
(`torch`, `torchaudio`, `transformers`, `datasets`, `mlflow`, …). On a
fresh install pip would resolve them to newer, mutually-incompatible
versions — the exact failure mode the nvidia notes warn about.

**Fix:** pin the macOS env to the same versions the nvidia env uses,
minus the CUDA-only bits (no `--extra-index-url …/cu130`, no `+cu130`
wheel tags — the plain `torch` wheel already targets Apple-Silicon
Metal/MPS). `mlx` / `mlx-lm` are intentionally left **un-pinned**: the
case-study models use a brand-new architecture (`qwen3_5`) that only the
latest `mlx-lm` understands. The pinned file is committed as
`environments/macos.yaml`; its header documents the rationale.

> If you ever see import errors or "model fails to load" after editing
> the env, re-check that `torch` and `torchaudio` share the same `2.10.x`
> minor and that `transformers==5.2.0`.

### Step 1 — create and verify the env

```bash
conda env remove -n benchmark_macos -y      # only if an old one exists
conda env create -f environments/macos.yaml
conda activate benchmark_macos
```

Sanity check (all should import, MPS should be available):

```bash
python - <<'PY'
import torch, transformers, mlx.core, mlx_lm, chromadb, datasets, radt, mlflow
print("torch", torch.__version__, "| mps:", torch.backends.mps.is_available())
print("transformers", transformers.__version__, "| mlx_lm OK")
PY
```

### Hurdle 3 — model names look wrong but are real

The MLX configs reference `mlx-community/Qwen3.5-9B-OptiQ-4bit` and
`mlx-community/Qwen3.5-4B-OptiQ-4bit`. These look like typos (Qwen *3.5*?
"OptiQ"?) but **they are real** HuggingFace repos (verify via
`https://huggingface.co/api/models/<id>`). They download automatically on
first run. Note: a plain `curl` to `huggingface.co/<id>` returns HTTP 200
even for non-existent repos (soft-404) — use the **`/api/models/`**
endpoint to truly check existence.

### Step 2 — run the two pipelines

```bash
python main.py evaluation/self_rag/configs/factoid_monolith_mlx.yml   -p 0 --label self_rag_monolith
python main.py evaluation/self_rag/configs/factoid_decomposed_mlx.yml -p 0 --label self_rag_decomposed
```

What each run does (verified from the live logs):
- `SelfRAGDataLoader` loads 918 QA pairs from `rag-datasets/rag-mini-wikipedia`.
- `ChromaRetriever` indexes 3 200 passages into an **in-memory** ChromaDB
  collection (first run downloads ChromaDB's default ONNX MiniLM embedder).
- The LLM stage(s) load the MLX model(s) and generate; routers parse the
  output and drive the Self-RAG retry/rewrite loop (`max_retries: 2`).
- `TerminalCapture` appends one JSON record per finished query.

The configs run **`max_queries: 10`** at Poisson **`rate: 1.0`** (the
committed MLX defaults — note this is 10, not the "5" the smoke-test prose
in `evaluation/self_rag/README.md` mentions, and not the 30 the CUDA
configs use).

### Hurdle 4 — `print()` output is block-buffered to the log file

When you redirect a run to a file, the framework's `print()` progress
lines are block-buffered and appear in big bursts (the `mlflow`/`logging`
lines are not). It can look stalled when it isn't. Two ways to see live
progress:
- prepend **`PYTHONUNBUFFERED=1`** to the command, or
- watch the result files grow instead of stdout:
  `wc -l evaluation/results/self_rag_monolith_outputs.jsonl`.

### Hurdle 5 — every query shared one `query_id` (a real bug, now fixed)

**Symptom:** in the `_outputs.jsonl`, the `query_id` field is identical
across all records within a run — and *different* between runs (the
monolith run was all `8b421854-…`, the decomposed run all `86d91487-…`).

**Root cause:** `utils/schemas/query.py` declared the field as

```python
query_id: int = uuid.uuid4()        # evaluated ONCE at import time
```

A dataclass default expression is evaluated a single time, when the class
is defined — so one UUID is generated at import and baked in as *the*
default. Every `Query(...)` built without an explicit id (the schedulers
never pass one — they only set `batch` and `out_flow_id`) gets that same
constant. A different constant per process is the tell-tale sign.

**Why it's not just cosmetic.** Scoring is fine (the verifier keys on
question text, and `batch` / `out_flow_id` are unique per query). But the
routers (`MonolithRouter`, `BinaryRouter`) track the Self-RAG retry budget
in a dict keyed by `query_id`:

```python
if query_id not in self._query_retries:
    self._query_retries[query_id] = self._max_retries
self._query_retries[query_id] -= 1
```

With a constant id that dict holds **one entry shared by all queries**, so
`max_retries: 2` becomes "2 retries total for the whole run" instead of
per-query. Once exhausted, later failing queries skip their rewrite/retry
loop and go straight to the error end-state.

**Fix (applied):** use a factory so each instance gets a fresh id (and
correct the bogus `int` type hint):

```python
from dataclasses import dataclass, field
query_id: uuid.UUID = field(default_factory=uuid.uuid4)
```

After the fix the JSONL shows 10 distinct `query_id`s per run, and each
query gets its own independent retry budget.

### Hurdle 6 — slow / lingering shutdown is expected

After "Pipeline execution completed in N seconds", the process can take
extra time tearing down (MLflow telemetry sockets, ChromaDB/embedder
semaphores, MLX Metal teardown). `main.py` deliberately calls `os._exit(0)`
after flushing traces to cut this short. You may see a benign
`resource_tracker: There appear to be 1 leaked semaphore objects` warning
— harmless; all results are already on disk by then. Each run exits 0.

### Step 3 — score the runs

```bash
python evaluation/scripts/verify_complex_cases.py self_rag_monolith self_rag_decomposed
```

This reads the two `_outputs.jsonl` files and writes
`evaluation/results/verification_report.md` with per-pipeline
golden-answer-hit counts and a monolith-vs-decomposed parity section.

### Hurdle 7 — decomposed factoid OOMs the GPU at the default (pipelined) setting

This is the one real failure on a 16 GB Mac, and it's worth understanding.

`serialize_queries` defaults to **`False`** (pipelined) in
`utils/schemas/pipeline.py`, and the MLX configs set Poisson `rate: 1.0`
with `queue_depth: 50` — so multiple queries are deliberately in flight at
once. The two topologies react very differently:

- **Monolith** — a single 9B model behind one mutex. Only one generation
  runs at a time regardless of how many queries are queued, so peak memory
  is bounded. It completes 10/10 pipelined.
- **Decomposed** — three *separate* 4B model instances (grader, generator,
  hallucination-checker) that can run **concurrently** across in-flight
  queries, each holding its own KV cache. On a 16 GB machine the combined
  weights (~3× 4B) plus stacked KV caches blow past the MPS allocation:

  ```
  libc++abi: terminating due to uncaught exception of type std::runtime_error:
  [METAL] Command buffer execution failed: Insufficient Memory
  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
  ```

  In my run it crashed (exit 134 / SIGABRT) after ~5 of 10 queries.

**Fix (used in the TL;DR):** run the decomposed pipeline with
**`--serialize true`**. That forces one query in flight at a time, so only
one of the three models is generating at any moment — KV caches don't
stack and it completes 10/10. This is the right choice when you only care
about answer quality (the verifier).

**Alternative (if you specifically need the pipelined cell** for the 2×2
timing sweep in `monolith_vs_decomposed_m2pro.md`): keep it pipelined but
lower the load — copy the config to a temp file and set `rate: 0.4` and
`max_queries: 8`, as the authors document in that file's "Caveat" section.

On a large-memory box (e.g. the 128 GB DGX Spark) this constraint
disappears and the default pipelined run is fine.

---

## Results I got (reproduced on this 16 GB Apple-Silicon Mac)

Both factoid pipelines ran end-to-end over the 10 `rag-mini-wikipedia`
questions. Quality (from `verification_report.md`):

| Pipeline | Wall time | Answered | Golden-answer hits | Notes |
|---|---|---:|---:|---|
| **Monolith** (1× 9B), pipelined  | 156.9 s | 9 / 10 | **8 / 10** | query 10 exhausted retries (retrieval miss) |
| **Decomposed** (3× 4B), serial   | 215.5 s | 8 / 10 | **7 / 10** | queries 3 & 10 exhausted retries (retrieval miss) |

Cross-pipeline parity: 8 shared questions answered by both, avg Jaccard
0.55 — moderate overlap, as expected for different model sizes.

**Verdict reproduced:** answer quality is a near-tie (8 vs 7 golden hits,
all genuine answers on-topic and grounded), exactly the qualitative
conclusion of `monolith_vs_decomposed_m2pro.md` (which reports monolith
8/10 hits, decomposed 7/8). Two notes when reading the table:

- The wall times are **not** an apples-to-apples speed comparison — the
  monolith ran *pipelined* (its default) and the decomposed ran *serial*
  (forced, to dodge the OOM in Hurdle 7). For the proper 2×2 speed sweep
  use the full procedure in `monolith_vs_decomposed_m2pro.md`.
- The verifier's golden-hit test is a strict substring match, so a couple
  of correct answers score as misses (e.g. "What did the Legal Tender Act
  of 1862 establish?" — both pipelines answer correctly but the long
  golden string "…the first paper currency in United States history"
  isn't a verbatim substring). Real answer quality is a touch higher than
  the hit counts suggest.

Per-run artifacts in `evaluation/results/`:
`self_rag_monolith.csv` / `_outputs.jsonl`,
`self_rag_decomposed.csv` / `_outputs.jsonl`,
and the combined `verification_report.md`.

---

## Optional — the full 2×2 contention sweep

The headline M2 Pro numbers in `monolith_vs_decomposed_m2pro.md` come from
running each topology both **pipelined** (`--serialize false`) and
**serial** (`--serialize true`), then running the bandwidth analyzer. See
that file's "Reproduce" block. On a 16 GB machine the
**decomposed + pipelined** cell can hit `[METAL] Insufficient Memory`
(3× 4B models × in-flight KV caches); the documented workaround is to
re-run just that cell with `rate: 0.4` and `max_queries: 8`.

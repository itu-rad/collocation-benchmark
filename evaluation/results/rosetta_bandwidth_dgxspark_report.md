# Rosetta RAG bandwidth-contention 2×2 — DGX Spark (GB10)

**Hardware:** NVIDIA DGX Spark, GB10 Grace Blackwell, 128 GB unified LPDDR5x
(~273 GB/s practical), single GPU + ARM CPU.
**Stack:** PyTorch 2.10.0+cu130, transformers 5.2, BF16 (no quantization).
**Models:** T1 = `Qwen/Qwen3.5-9B` monolith · T2 = 3× `Qwen/Qwen3.5-4B`
distributed (grader / generator / hallucination; rewriter shares the grader
model via `depends_on_id`).
**Load:** PoissonLoadScheduler rate 4.0, `max_queries: 30`, `queue_depth: 50`,
greedy decoding (`do_sample: false`). 30 queries per cell.
**GPU residency verified:** `nvidia-smi pmon` showed the inference process at
**73–74 % SM** during generation; `dcgmi -e 203` GPUTL ~96 % under load. The HF
stage runs genuinely on GB10, not a CPU fallback. (`nvidia-smi --query-gpu`
util/mem read `N/A` on GB10's unified memory — use `pmon`/`dcgmi`.)

This ports the macOS M2 Pro VQA bandwidth-contention experiment to the Rosetta
RAG pipelines on CUDA. **Mapping axis** A/B is reinterpreted as topology:
A = T1 monolith, B = T2 distributed. **Schedule axis**: pipelined
(`serialize_queries=false`, many queries in flight) vs serial
(`serialize_queries=true`, one query end-to-end at a time).

## (a) The 2×2

| Cell | Wall (s) | Throughput (q/s) | Mean latency (s) | median / p95 (s) | Stage concurrency mean / max | Device busy |
|------|---------:|-----------------:|-----------------:|-----------------:|:----------------------------:|-------------|
| **T1 monolith — pipelined** | 260.40 | 0.115 | 147.12 | 156.4 / 245.0 | **1.07** / 3 | cuda 100 %, CPU 1 % |
| **T1 monolith — serial**    | 262.88 | 0.114 | 8.76   | 8.78 / 13.4   | **0.99** / 1 | cuda 98 %, CPU 1 % |
| **T2 distributed — pipelined** | 140.36 | 0.214 | 72.67 | 74.7 / 123.0 | **1.49** / 5 | cuda 100 %, CPU 3 % |
| **T2 distributed — serial**    | 154.42 | 0.194 | 5.15  | 4.32 / 7.0   | **0.97** / 1 | cuda 95 %, CPU 2 % |

Answer quality was effectively constant across all four cells (golden-answer
hits 20–21 / 30; 25–28 / 30 answered), confirming that scheduling changed only
*timing*, not correctness, and matching the Mac evidence's hit pattern.

**Pairwise reads**

- *Pipelining ON vs OFF (same topology):* wall time barely moves —
  **T1: −0.9 %**, **T2: −9.1 %** — while mean per-query latency inflates
  **16.8× (T1)** and **14.1× (T2)**. Throughput gain from pipelining is
  **+1.0 % (T1)** and **+10.0 % (T2)**.
- *Decomposition advantage (T2 vs T1):* T2 is faster in both regimes —
  serial **−41.3 %** latency / **+70.2 %** throughput; pipelined **−50.6 %**
  latency / **+85.5 %** throughput.

## (b) Heterogeneity-advantage collapse vs M2 Pro

The signature on M2 Pro (VQA, CLIP→FAISS→LLM): the heterogeneous mapping's
advantage **collapsed** under pipelining because CLIP-on-ANE and LLM-on-MPS are
distinct engines that run concurrently and then contend for the one LPDDR bus.

| Platform / experiment | Advantage **serial** | Advantage **pipelined** | Collapse (serial − pipe) |
|---|---:|---:|---:|
| M2 Pro — VQA (mapping A vs B) | +4.5 % | −7.9 % | **+12.4 pp** (collapses) |
| DGX Spark — Rosetta (T1 vs T2, latency) | +41.3 % | +50.6 % | **−9.3 pp** (grows) |
| DGX Spark — Rosetta (T1 vs T2, throughput) | +70.2 % | +85.5 % | **−15.3 pp** (grows) |

On DGX Spark the decomposition advantage **does not collapse** under
contention — it widens. There is no bandwidth-bound advantage-collapse
signature here.

## (c) Was the contention regime reached at rate 4.0?

Only weakly, and **raising the rate would not help** — for an architectural
reason, not a tuning one:

- Every stage runs on the **single** GB10 GPU, and each `Inference` stage holds
  a per-model mutex around `generate()`. T1 has one model (monolith + rewriter
  share it via `depends_on_id`), so **all** GPU work is serialized through one
  lock → stage-concurrency mean **1.07**, max 3 (the surplus is CPU
  retriever/formatter overlapping the lone GPU generate). T2 has three
  independent 4B instances → up to 3 generates can interleave → concurrency
  mean **1.49**, max 5.
- The GPU is **~100 % busy in every cell, serial and pipelined alike.** Under
  pipelining the 30 queries are admitted in ~7.5 s (rate 4.0) and then queue
  behind the saturated GPU — that queue *is* the 14–17× latency inflation,
  while wall time stays essentially flat. The system is already maximally
  loaded; a higher arrival rate only deepens the queue, it cannot create
  additional concurrent execution on one accelerator behind a mutex. Rate was
  therefore left at 4.0.

## (d) Verdict

**DGX Spark does *not* reproduce the bandwidth-bound advantage-collapse
signature for the Rosetta decomposition — but the reason is architectural, not
"GB10's bandwidth is so high it escaped the regime."** The M2 Pro signature
depends on two *distinct* accelerators (ANE + GPU) executing concurrently and
contending for one memory bus; the Rosetta-on-CUDA port collapses both
"mappings" onto a single GB10 GPU, so there is no second engine for stages to
spread across and thus no cross-engine shared-bandwidth contention to expose.
What the 2×2 instead measures is monolith-vs-decomposed *on one saturated
accelerator*: the GPU runs at ~100 % in all cells, pipelining buys almost no
throughput (+1 % T1, +10 % T2) and only inflates latency via queueing, and the
3×4B decomposition beats the 9B monolith in every regime — an advantage that
*grows* under load (the opposite of a collapse) because T2's three independent
model instances admit a little real overlap (concurrency 1.49 vs T1's 1.07)
whereas the monolith's single mutex hard-serializes it. To actually probe the
unified-memory bandwidth thesis on GB10 one would need a genuinely
heterogeneous mapping — e.g. an encoder/retriever on a second engine
(NVDLA/CPU/DSP) concurrent with LLM decode on the GPU — so that two
simultaneously-active consumers contend for the ~273 GB/s LPDDR5x. As ported,
the Rosetta experiment sits in a single-accelerator, mutex-serialized,
queueing-dominated regime where the bandwidth-contention question is not
testable.

---
*Generated by `evaluation/scripts/bandwidth_analysis.py` (machine report:
`rosetta_bandwidth_report.md`); cells `rosetta_t{1,2}_{pipe,serial}`.*

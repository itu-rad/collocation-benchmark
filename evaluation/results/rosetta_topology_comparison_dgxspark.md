# Rosetta RAG: monolith vs decomposition — DGX Spark (GB10)

**Question.** For an agentic RAG pipeline, is it better to use **one large model
that does everything** (relevance grade + answer + hallucination check in a
single call) or **several smaller specialized models**, one per sub-task? This
compares the two topologies on identical work.

**Hardware:** NVIDIA DGX Spark, GB10 Grace Blackwell, 128 GB unified LPDDR5x,
single GPU. **Stack:** PyTorch 2.10.0+cu130, transformers 5.2, BF16 (no quant).
**GPU residency verified** (`nvidia-smi pmon`: 73–74 % process SM during
generation — genuinely on GB10, not CPU).

| Topology | Models | Per query |
|---|---|---|
| **T1 — Monolith** | 1× `Qwen/Qwen3.5-9B` (rewriter shares it) | one 256-tok structured call that grades + answers + self-checks |
| **T2 — Decomposed** | 3× `Qwen/Qwen3.5-4B` (grader · generator · hallucination; rewriter shares the grader) | grade (16 tok) → generate (256 tok) → hallucination-check (16 tok) → optional rewrite (128 tok) + retry |

Workload: 30 questions from `rag-mini-wikipedia`, top-3 ChromaDB retrieval,
self-RAG retry loop (max 2). Each topology run both pipelined
(`serialize_queries=false`) and serial (`true`).

## Headline comparison

| Metric | T1 Monolith (9B) | T2 Decomposed (3×4B) | Winner |
|---|---:|---:|:---:|
| Per-query latency (serial) | 8.76 s | **5.15 s** (−41 %) | T2 |
| Throughput (pipelined) | 0.115 q/s | **0.214 q/s** (+86 %) | T2 |
| Throughput (serial) | 0.114 q/s | **0.194 q/s** (+70 %) | T2 |
| Wall time, 30 q (pipe / serial) | 260 / 263 s | **140 / 154 s** | T2 |
| Golden-answer hits | 20 / 30 | **20–21 / 30** | tie |
| Answered (not retry-exhausted) | 25 / 30 | **27–28 / 30** | T2 |
| LLM calls per query | **1.13** | 3.1 | T1 |
| Model memory (BF16) | **~18 GB** | ~24 GB | T1 |
| Distinct models to operate | **1** | 3 | T1 |

Answer **quality is a tie** (same golden-hit rate), and decomposition actually
*answered slightly more* questions (fewer hit the retry ceiling). So at this
task difficulty the smaller models lose nothing on accuracy.

## Where the wall-clock time goes (per-stage time-share)

This is the mechanism behind the result — measured from the run traces:

**T1 Monolith (pipelined):**
| Stage | runs | busy | % wall |
|---|---:|---:|---:|
| Monolith LLM (9B, 256 tok) | 32 | 260.2 s | **100 %** |
| Query-rewrite LLM (retries) | 2 | 13.8 s | 5 % |
| all CPU stages (retrieve/format/route) | — | 3.5 s | ~1 % |

**T2 Decomposed (pipelined):**
| Stage | runs | busy | % wall |
|---|---:|---:|---:|
| Generator LLM (4B, 256 tok) | 30 | 138.1 s | **98 %** |
| Grader LLM (4B, 16 tok) | 32 | 31.1 s | 22 % |
| Hallucination LLM (4B, 16 tok) | 30 | 29.6 s | 21 % |
| Query-rewrite LLM (retries) | 2 | 5.9 s | 4 % |
| all CPU stages | — | 3.6 s | ~2 % |

Reading: **both topologies are bottlenecked on the same thing — generating the
256-token answer.** The monolith does it on a 9B (≈260 s of GPU time); the
decomposed pipeline does it on a 4B generator (≈138 s). The grader and
hallucination checks are *cheap* (16 tokens each, ~30 s total) and in pipelined
mode they overlap the generator rather than adding to wall time (busy %'s sum
past 100 % because they run concurrently). So decomposition does **2.8× more LLM
calls per query (3.1 vs 1.13)** yet finishes ~45 % sooner, because the dominant
cost moved from a 9B to a 4B and the extra calls are tiny and overlap-able.

## Behaviour under load (pipelined vs serial)

| | T1 Monolith | T2 Decomposed |
|---|---:|---:|
| Stage concurrency, mean / max (pipe) | 1.07 / 3 | **1.49 / 5** |
| Throughput gain from pipelining | +1.0 % | **+10.0 %** |
| Latency inflation under pipelining | 16.8× | 14.1× |

The monolith is a **single model behind one mutex**, so concurrent queries
can't overlap on the GPU at all — pipelining adds nothing but queue (throughput
+1 %). The decomposed pipeline has **three independent model instances**, so a
grader/checker call for one query can run alongside the generator for another
(concurrency 1.49 vs 1.07), giving it a real, if modest, throughput edge under
load (+10 %). On a single GPU both regimes still saturate the device
(~100 % busy), so the latency blow-up under pipelining is pure queueing in both
cases — if you care about per-query latency, run serial; if about throughput,
decomposition pipelines better.

## Trade-off summary

**Decomposition (3×4B) — wins on performance:**
- ~41 % lower per-query latency, ~70–86 % higher throughput, ~45 % less wall time.
- Equal answer quality and slightly better coverage at this task difficulty.
- Each sub-task is right-sized; stages can be tuned/swapped independently; pipelines better under load (independent models overlap).

**Decomposition — costs:**
- ~33 % more model memory (~24 GB vs ~18 GB; three resident 4B instances).
- 2.8× more LLM invocations per query → more orchestration, more failure surface, retries amplified across stages.
- More moving parts (routers, formatters, retry loops) to build and debug.

**Monolith (9B) — wins on simplicity:**
- One model, one code path, ~25 % less memory, fewest invocations, easiest to operate and attribute failures.
- But pays in latency/throughput (the 9B is the sole bottleneck) with no quality return here.

## Verdict

**For this factoid-RAG workload on GB10, decomposition into smaller specialized
models is the better topology:** it is markedly faster (latency and throughput)
at equal answer quality, because the expensive answer generation runs on a 4B
instead of a 9B and the added grade/check calls are cheap and overlap-able. The
monolith's only real advantage is operational simplicity and a smaller memory
footprint. The conclusion is task-dependent in one important way: the sub-tasks
here (binary relevance grading, factoid answering, yes/no hallucination check)
are easy enough that 4B matches 9B on accuracy — on harder reasoning where the
9B's extra capacity would lift quality, the calculus could flip toward the
monolith (or toward decomposing into *larger* specialists). Two further notes:
(1) decomposition's throughput edge comes from running independent models
concurrently, so it grows with more hardware parallelism and shrinks toward the
monolith on a single saturated accelerator; (2) a fairer capacity-matched
comparison would pit the 9B monolith against a 9B-class generator in the
decomposed pipeline, isolating "specialization" from "smaller model."

---
*Cells `rosetta_t{1,2}_{pipe,serial}` (30 q each); per-cell timing also in the
machine report `rosetta_bandwidth_report.md`. Per-stage counts/time-share
extracted from the run traces via `evaluation/scripts/bandwidth_analysis.py`'s
parser.*

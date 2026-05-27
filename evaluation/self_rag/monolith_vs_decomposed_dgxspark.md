# Self-RAG: monolith vs decomposition — DGX Spark (GB10)

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
| **Monolith** | 1× `Qwen/Qwen3.5-9B` (rewriter shares it) | one 256-tok structured call that grades + answers + self-checks |
| **Decomposed** | 3× `Qwen/Qwen3.5-4B` (grader · generator · hallucination; rewriter shares the grader) | grade (16 tok) → generate (256 tok) → hallucination-check (16 tok) → optional rewrite (128 tok) + retry |

Workload: 30 questions from `rag-mini-wikipedia`, top-3 ChromaDB retrieval,
self-RAG retry loop (max 2). Each topology run both pipelined
(`serialize_queries=false`) and serial (`true`).

## Headline comparison

| Metric | Monolith (9B) | Decomposed (3×4B) | Winner |
|---|---:|---:|:---:|
| Per-query latency (serial) | 8.76 s | **5.15 s** (−41 %) | Decomp |
| Throughput (pipelined) | 0.115 q/s | **0.214 q/s** (+86 %) | Decomp |
| Throughput (serial) | 0.114 q/s | **0.194 q/s** (+70 %) | Decomp |
| Wall time, 30 q (pipe / serial) | 260 / 263 s | **140 / 154 s** | Decomp |
| Golden-answer hits | 20 / 30 | **20–21 / 30** | tie |
| Answered (not retry-exhausted) | 25 / 30 | **27–28 / 30** | Decomp |
| LLM calls per query | **1.13** | 3.1 | Mono |
| Model memory (BF16) | **~18 GB** | ~24 GB | Mono |
| Distinct models to operate | **1** | 3 | Mono |

Answer **quality is a tie** (same golden-hit rate), and decomposition actually
*answered slightly more* questions (fewer hit the retry ceiling). So at this
task difficulty the smaller models lose nothing on accuracy.

## Where the wall-clock time goes (per-stage time-share)

This is the mechanism behind the result — measured from the run traces:

**Monolith (pipelined):**
| Stage | runs | busy | % wall |
|---|---:|---:|---:|
| Monolith LLM (9B, 256 tok) | 32 | 260.2 s | **100 %** |
| Query-rewrite LLM (retries) | 2 | 13.8 s | 5 % |
| all CPU stages (retrieve/format/route) | — | 3.5 s | ~1 % |

**Decomposed (pipelined):**
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

| | Monolith | Decomposed |
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

## Difficulty escalation: multi-hop (HotpotQA)

The factoid result above is a tie on quality, so the obvious follow-up: on a
*harder* task that needs multi-step reasoning, does the 9B monolith's extra
capacity finally beat the 3×4B decomposition? We re-ran both topologies on
**HotpotQA** (FlashRAG `hotpotqa` dev, 30 q), with the Chroma corpus built from
each question's gold + distractor contexts (~3k passages) and `top_k: 5` so the
supporting passages are *retrievable but buried*. Same models, same pipelines.

| HotpotQA multi-hop | Answered | Golden hits | Serial latency | Throughput (pipe) | Wall (pipe) |
|---|---:|---:|---:|---:|---:|
| **Monolith (9B)** | 10 / 30 | **5 / 30 (17 %)** | 6.32 s | 0.160 q/s | 187.4 s |
| **Decomposed (3×4B)** | **20 / 30** | **10 / 30 (33 %)** | **4.75 s** | **0.199 q/s** | **150.4 s** |

**The harder task did not flip the verdict — it widened the decomposition
lead.** Decomposition scored **2× the golden hits** (33 % vs 17 %), answered
**2× as many** questions, *and* stayed faster (−25 % serial latency, +24 %
throughput). The capacity hypothesis is not supported here.

**Why the monolith loses on hard inputs (from the traces):** the monolith does
relevance-grade + answer + hallucination-check in **one forced-JSON call**, and
on multi-hop questions it becomes brittle — it grades the retrieved context
"not relevant" and bails. It hit *"no satisfactory answer after retries"* on
**19 / 30** questions (vs answering only 10). The decomposed pipeline's
**dedicated 4B relevance grader** — a narrow yes/no prompt — is more robust at
that one sub-task, so it passed 20/30 through to a dedicated answerer. The
benefit of decomposition is therefore **specialization/robustness, not just
speed**, and it *grows* with task difficulty: a small model doing one narrow job
well beats a larger model juggling three jobs in a single structured pass.

Easy → hard summary (golden hits / 30):

| | Monolith 9B | Decomposed 3×4B |
|---|---:|---:|
| Factoid (rag-mini-wikipedia) | 20 | 20–21 |
| Multi-hop (HotpotQA) | **5** | **10** |

**Caveats.** (1) Absolute scores are low — multi-hop RAG is genuinely hard for
4–9B models and the golden-hit metric is strict substring overlap. (2) Single-
shot `top_k: 5` retrieval caps recall on 2-hop questions, so scores are partly
retrieval-limited (the self-RAG rewrite loop only partly compensates). (3) This
compares "one model, *combined* task" vs "three models, *split* tasks" — it
conflates specialization with prompt decomposition. A cleaner control would give
the monolith three *separate* calls (grade, then answer, then check) on the same
9B, or pit the 9B monolith against 9B-class specialists, to isolate whether the
win is from smaller-but-focused models or simply from not cramming three jobs
into one JSON. That control is the recommended next step.

### Follow-up: context-aware (multi-hop) query rewrite + evidence accumulation

The first multi-hop run used the stock self-RAG rewriter, which only
*paraphrases the original question* and never sees the retrieved documents, and
a retriever that *replaces* documents each hop. That can't bridge — e.g. for
*"The director of Big Stone Gap is based in what NY city?"* hop 1 finds the
film (→ director Adriana Trigiani) but a paraphrase just re-retrieves the same
page. So we made the rewriter **condition on the accumulated documents** (emit a
focused follow-up/bridge query) and the retriever **accumulate evidence across
hops** while keeping grading/answering anchored to the user's true question.

The mechanism verifiably works: for *"Were Scott Derrickson and Ed Wood of the
same nationality?"* the rewriter now emits the bridge query
**"Scott Derrickson nationality"**, and documents accumulate 5 → 8–10 across
hops. **But end-to-end golden hits did not move:**

| HotpotQA | context-blind rewrite | context-aware + accumulation |
|---|---:|---:|
| Monolith (9B) | 5 / 30 | 5 / 30 |
| Decomposed (3×4B) | 10 / 30 | 10 / 30 |

**Why the fix didn't help (quantified from the traces):** the second hop fired
on only **~1–2 of 30 questions** (just 1/30 accumulated >5 docs). The self-RAG
control flow only triggers a rewrite/re-retrieval when the grader scores the
hop-1 documents *"not relevant"* — but for multi-hop questions those documents
are *partially* relevant (they cover one of the two entities), so the grader
says "yes" and the pipeline **answers prematurely**, before any bridge query can
fire. Context-aware rewriting is therefore **necessary but not sufficient**: the
binding constraint is the control flow, not the rewrite prompt. A relevance-
gated retry loop is not a multi-hop planner.

**What would actually move multi-hop accuracy** (recommended next step): a
control flow that performs multi-hop deliberately rather than as a fallback —
e.g. decompose the question into sub-questions up front, or always run ≥2
retrieval hops (Self-Ask / IRCoT proper) so the bridge query and accumulated
evidence are *used on every question*, not just the ~5 % the grader happens to
reject. The decomposition-vs-monolith verdict is unchanged by this (decomposition
still doubles the monolith).

---
*Factoid cells `self_rag_factoid_{monolith,decomposed}_{pipe,serial}`; multi-hop
cells `self_rag_multihop_{monolith,decomposed}_{pipe,serial}`, 30 q each (configs
in `evaluation/self_rag/configs/`). The two multi-hop rows above (context-blind
vs context-aware rewrite) are successive code states of the same configs.
Per-stage counts/time-share extracted from the run traces via
`evaluation/scripts/bandwidth_analysis.py`'s parser.*

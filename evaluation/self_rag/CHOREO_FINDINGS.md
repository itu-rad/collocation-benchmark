# Self-RAG — What Choreo Surfaces About Agentic-RAG Cost and Control Flow

> **E4/E5 FRAMING SUPERSEDED (2026-09-02).** This document predates the agreed §5 case-study
> plan. Where it describes E4 as a "prefill/decode flip" or E5 as an indexer study, it is stale:
> **§5.1 is Self-RAG execution strategies** (monolithic prompt / shared+locked / per-role copies /
> server continuous-batching) framed *investigatively* — narrowing down what decomposition causes —
> and **§5.2 is collocation TYPES with per-pipeline attribution**, where the background workload is
> a prop. The predictive cost law and the flip thesis are NOT carried forward. Authoritative:
> `EXPERIMENTS.md` (E4/E5 sections). Everything else in this file still applies.


*Choreo evaluation. All numbers below are from the **strengthened-retriever re-run** (2026-07-23):
retriever = `intfloat/e5-base-v2` (MLPerf `e2e-rag`'s embedder) at top_k=5, replacing ChromaDB's weak
default (`all-MiniLM-L6-v2`, top_k=3) which had capped accuracy by mis-ranking supporting passages
(see §R). Accuracy is scored three ways — exact-match and token-F1 (project scorer, `_best_answer`)
plus an **LLM judge** (Haiku, semantic factual equivalence; verdicts under `evaluation/self_rag/judge/`)
— because exact-match materially mis-ranks these arms (§0). Greedy decoding → answers are byte-identical
across runs, so R=1 quality is exact. MiniLM-retriever baseline preserved in `_baseline_miniLM/`.*

## Setup — arms, models, and sizes

| Arm | Model | Structure |
|---|---|---|
| **monolith** | **Qwen3.5-9B** (bf16 on cuda / 9B-OptiQ-4bit on mlx) | one 9B model, **single call** (relevance + answer + hallucination in one JSON) |
| **monolith_4b** | **Qwen3.5-4B** | one 4B model, single call (same monolithic prompt) |
| **decomposed** | **Qwen3.5-4B**, one instance per role | 4B, **separate call per role** (relevance grader, generator, hallucination grader, query-rewrite) |
| **decomposed_shared** | **Qwen3.5-4B**, one instance + mutex | 4B, the same role calls through one shared instance |

- **Devices:** cuda = GB10/DGX Spark (bf16); mlx = M2 Pro (4-bit OptiQ). Same retriever across arms
  (`e5-base-v2`, top_k=5); corpus is `rag-mini-wikipedia` for factoid and HotpotQA contexts for
  multihop; N=120 each.
- Two comparisons recur below: **structure at fixed size** (`decomposed` 4B vs `monolith_4b` 4B) and
  **decompose-a-small-model vs use-a-big-one** (`decomposed` 4B vs `monolith` 9B).

**Framing.** MLPerf recently standardized this workload (`e2e-rag`: iterative multi-hop retrieval +
LLM grading/decomposition), but benchmarks it **at the OpenAI-API boundary** (the LLM behind a
vLLM/OpenRouter endpoint), so its unit is request→response. Choreo runs the same structure
**in-process** and dissects it per-stage (prefill/decode at `first_token`) and per-attempt (the retry
trajectory). Everything below is what that dissection surfaces beneath the API boundary — stated as
tool capability, never as a critique of a benchmark that is correctly backend-agnostic.

---

## §0 Baseline — are the arms comparable, and how do they perform?

Before any per-stage number means anything, the arms must be **quality-comparable** and their
**end-to-end** cost established. Choreo gives all of this from the same runs (accuracy via the scorer;
latency/throughput from the per-query trace).

### Accuracy — and why exact-match alone is the wrong instrument here

Scored **three ways**: exact-match (EM), token-F1, and an **LLM judge** (Haiku, semantic factual
equivalence). EM/F1 are deterministic and reproducible (MLPerf-style); the judge is the semantic lens
that a lexical metric misses. Values are EM / F1 / **Judge**:

| task | monolith (9B) | monolith_4b (4B) | decomposed (4B) |
|---|---|---|---|
| **cuda factoid** | 0.492 / 0.544 / **0.808** | 0.667 / 0.741 / **0.842** | 0.675 / 0.754 / **0.842** |
| **cuda multihop** | 0.267 / 0.335 / **0.342** | 0.258 / 0.334 / **0.367** | 0.300 / 0.369 / **0.375** |
| **mlx factoid** | 0.508 / 0.560 / **0.792** | 0.675 / 0.744 / **0.842** | 0.633 / 0.707 / **0.808** |
| **mlx multihop** | 0.217 / 0.269 / **0.300** | 0.242 / 0.319 / **0.333** | 0.267 / 0.335 / **0.333** |

*(`decomposed_shared` omitted: byte-identical to `decomposed` under greedy.)*

**Under semantic judging the arms are quality-comparable — the EM ranking was a scoring artifact, not
a quality difference.** On cuda factoid the EM spread 0.492→0.675 **collapses to 0.808→0.842** by the
judge (all three within ~3 pp). The apparent "the 9B is worst" under EM is **exact-match penalizing
verbose answers**: the 9B answers a yes/no question with `"The Celsius crater on the Moon is named
after him"` (correct, EM=0) instead of the bare `"yes"`, so its EM deficit is entirely on the yes/no
half. Under the judge the 9B is neither best nor worst — it is simply *tied*, which is itself the key
result: **a 2.25×-larger model buys no measurable quality here** (see §R for why — the bottleneck is
retrieval, not the generator).

*Judge validity (the reproducibility guard, since an LLM judge is not itself reproducible):* across all
12 cells the judge **only ever rescues EM false-negatives — it overturned zero clean EM matches**
(one-directional, exactly the expected signature). One initially-corrupt cell (11 exact-match answers
wrongly failed) was caught by that same overturn check and re-judged. Rescues concentrate where EM is
lexically brittle — yes/no restatement, name variants (`"McClellan"` for `"General McClellan"`),
paraphrase, and **OCR-corrupted gold** (the source `rag-mini-wikipedia` gold contains typos like
`"MassachuS08_setts"` that EM/F1 punish and the judge reads through). Run-to-run judge noise is ≈±1–2
of 120 (~1 %). Verdicts persisted under `evaluation/self_rag/judge/`.

**Bottom line for the comparison:** all arms deliver comparable answer quality (if anything the 9B
edges ahead on the format-fair entity questions), so the cost/latency comparisons below are between
quality-equivalent systems — which is the only footing on which a cost comparison means anything.

### End-to-end performance (steady-state, warmup-excluded; single-stream)

| | monolith 9B | monolith_4b 4B | decomposed 4B |
|---|---|---|---|
| **cuda factoid** (p50 / p90 / q·s⁻¹) | 2.7 / 3.4 / 0.33 | **1.7 / 2.4 / 0.48** | 1.9 / 3.4 / 0.47 |
| **cuda multihop** | 2.7 / 10.4 / 0.19 | 2.1 / 7.1 / 0.26 | **2.0 / 4.2 / 0.37** |
| **mlx factoid** | 6.5 / 9.3 / 0.12 | **3.3 / 4.5 / 0.24** | 9.0 / 20.1 / 0.09 |
| **mlx multihop** | 7.7 / 34.9 / 0.06 | **3.7 / 17.8 / 0.13** | 9.4 / 22.1 / 0.08 |

**Reads that frame the whole section (and a correction the retriever upgrade forced):**
- **The performance winner is now `monolith_4b`, not `decomposed`** — at equal quality it has the lowest
  latency/highest throughput on 3 of 4 cells. With a *realistic* retriever (top_k=5, vs the old weak
  top_k=3), decomposition re-encodes 5 documents through each of its 3 roles, and that prefill cost
  erased the latency edge it showed under the weak retriever. **This is a case of Choreo's own
  measurement overturning a conclusion once the confound (a toy retriever) was removed** — the kind of
  correction the per-stage view makes visible.
- **`decomposed` still wins one thing on GB10: the multihop tail** (p90 4.2 s vs 7.1–10.4 s) — its short,
  fixed-length grader calls bound the tail where the monolith's variable generation does not.
- **Device dependence is now stark:** on the M2, `decomposed` is the *slowest* arm (factoid p50 9.0 s vs
  3.3 s for monolith_4b) — its 3× context re-encoding is punishing on the slow 4-bit device. The same
  topology that is competitive on GB10 is a poor choice on the M2. Decomposition's real payoff is **not
  latency** but token efficiency and control-flow behaviour (Findings 1–3).
- **`decomposed_shared` matches `decomposed` on quality/throughput at 1/3 the model memory** (one shared
  4B instance + mutex vs three).

---

## §R — The accuracy ceiling is *retrieval*, not the generator (and Choreo localizes it)

### 1) What Choreo offers
Choreo records the **retriever stage's output separately** from the LLM stage. So for every question it
can attribute a wrong answer to **retrieval** (the gold passage was not in what was retrieved) vs
**generation** (it was retrieved, the model still missed) — a decomposition a black-box QA score, or an
API-boundary RAG harness, structurally cannot make.

### 2) The finding: quality is retrieval-bound, so model size buys nothing
Splitting cuda entity questions by whether the gold answer is in the retrieved context (e5, top_k=5):

| | retrieval HIT | retrieval MISS | shared failures that are retrieval-misses |
|---|---|---|---|
| cuda factoid | 9B 0.72 / 4B 0.77 | 9B 0.06 / 4B 0.00 | 88 % |
| cuda multihop | 9B 0.61 / 4B 0.61 | ≈ 0 / ≈ 0 | 77 % |

**When retrieval succeeds, a 4B extracts the answer as well as the 9B; when it fails, neither can.** The
residual error budget is dominated by retrieval, so scaling the generator (2.25× the parameters) cannot
move the top-line number — and doesn't (§0: the 9B ties, never wins).

### 3) Robust to the obvious objection ("your retriever was weak")
We rebuilt the retriever — ChromaDB's default `all-MiniLM-L6-v2` (top_k=3) → **`e5-base-v2` (top_k=5)**,
the embedder MLPerf's `e2e-rag` uses — and re-ran everything. Entity retrieval-hit rose
**0.62→0.705 (factoid), 0.43→0.50 (multihop)**; judge accuracy rose **+8–11 pp** on factoid. **The
ceiling moved up, but the arms stayed comparable and the 9B still did not pull ahead on retrieval-hit
questions.** So "capacity doesn't help here" holds at *two* retriever strengths — it is a property of
the task, not an artifact of a toy retriever. And even the residual misses are mostly *ranking* failures
(the gold passage is in the corpus, just not top-k), which points the optimization at the **retriever**
(a reranker / stronger embedder), not a bigger LLM. That is the actionable, per-stage conclusion Choreo
delivers and a monolithic accuracy number cannot.

---

## Finding 1 — The compute in RAG is context re-encoding (prefill), most of it redundant

### 1) What Choreo offers
Because Choreo executes the pipeline as a **stage graph of independent threads**, we express the same
workload as four arms **by editing a YAML topology, not the code**. Every LLM stage is traced with a
**prefill/decode split at `first_token`**, giving per-role prefill time, decode time, and output-token
count automatically. And the **graph knows the grader/generator roles are siblings of one retrieval
node**, so it can attribute the *same retrieved-context object* as it is re-encoded across them — a
join a stock profiler (which sees three unrelated forward passes) cannot make.

### 2) The bigger picture
Where does the compute in an agentic-RAG pipeline actually go, and how does it move as you change the
pipeline's shape? Not "how fast is the model," but "which *phase* of which *role* dominates, and is
that a property of the model or of the topology?"

### 3) The findings (verified, quality cells)
- **Decomposition shifts the pipeline from decode-bound to prefill-bound — and it is topology, not
  model size.** Prefill's share of LLM time (cuda factoid): `monolith` (9B) **0.24**, `monolith_4b` (4B)
  **0.27** — both decode-leaning — vs `decomposed` (4B) **0.65**. The **fixed-size** step
  `monolith_4b (4B) 0.27 → decomposed (4B) 0.65` proves it is the *split*, not the 9B→4B change.
  Multihop: 0.30 / 0.33 → 0.63. On mlx the profile is already prefill-heavy and decomposition drives it
  to the wall: factoid 0.81 / 0.84 → **0.97**; multihop 0.84 / 0.87 → **0.97** (nearly *all* of
  decomposed's compute on the M2 is re-encoding context). *(These fractions rose vs the weak-retriever
  run because top_k=5 feeds more context to encode — the prefill effect is now larger.)* The industry's
  headline serving fix — decode-side continuous batching (Orca/vLLM) — targets the phase decomposition
  just shrank.
- **Output-token count barely predicts a role's cost, because prefill dominates.** Per call (cuda
  factoid decomposed): the relevance grader emits **4 tokens** and costs **560 ms**; the generator
  emits **6.1 tokens** and costs **623 ms** — so the grader is **90%** of the generator's per-call cost
  despite emitting a third fewer tokens (hallucination grader 83%). On mlx the effect is extreme — the
  graders cost **93–191%** of the generator per call (the 4-bit device makes re-encoding the 5-document
  context so expensive it *exceeds* the generator's decode). The cheap-output graders are expensive
  because each **re-encodes the full retrieved context** (prefill), which dwarfs the 2–6 decoded tokens.
- **~62% of the decomposed prefill is the same documents re-encoded 3×.** The three doc-bearing roles
  (grader, generator, hallucination) account for the large majority of all prefill and each re-encodes
  the identical retrieved documents, so two of every three copies are pure recompute — and top_k=5 makes
  each copy larger, so the absolute waste grew vs the weak-retriever run. **Sharing the model does not
  reclaim it:** `decomposed_shared`'s per-role prefill and prefill-fraction (0.65) are identical to
  `decomposed`'s — the mutex shares weights, not work. And the deployed prefix cache (vLLM
  APC / RadixAttention) **misses it by construction**: the roles share no common prefix (different
  preambles, wrapper strings, and docs-first vs docs-after-question order), so the byte-identical span
  sits at a different absolute position in each. Only a graph-level, position-independent hoist
  (PromptCache-style) removes it — **losslessly** (same tokens → same greedy output). *(This answers
  the report's open B.2: the decomposition gain does not survive prefix caching — the cache can't see
  the redundancy.)*

*Choreo capabilities used: graph modularity (sibling attribution), per-stage prefill/decode tracing,
in-process execution, residency (the shared-model arm).*

---

## Finding 2 — A 4B role-graph matches a 9B model at a fraction of the tokens — but "fewer tokens" ≠ "less energy"

### 1) What Choreo offers
The one-line topology switch gives `monolith` (9B), `monolith_4b` (4B) and `decomposed` (4B) at matched
work, and Choreo records — for free on every run — per-stage **generated tokens** (decode work), the
**prefill/decode split** (where the compute is), and, via the radt power listeners (`nvidia-smi` on
GB10, `macmon` on M2), **actual power/energy**. Joined to the answer scorer, these give *cost per
correct answer* in three currencies (tokens, wall-seconds, joules), decomposed into prefill vs decode
— a full efficiency ledger as a byproduct of running the arms.

### 2) The bigger picture
Buy a unit of accuracy two ways — reorganize a small model into roles, or scale the model up — and ask
which is cheaper per correct answer, and *in what currency*.

### 3) The findings (verified)
**Accuracy (semantic judge — see §0):** all arms are quality-comparable (cuda factoid 0.808 / 0.842 /
0.842 for 9B / monolith_4b / decomposed; the exact-match ranking was a formatting artifact). So "correct"
below is **judge-defined**, and the ledger is *not* distorted by exact-match's lexical penalty against
verbose answers — which matters, because that penalty fell hardest on the very arm (9B) whose
per-correct cost we are dividing.

**Decode tokens per correct answer** (robust — this is generation *length*, independent of the scorer):

| | decomposed(4B) | monolith_4b(4B) | monolith(9B) | dec vs 4B (structure) | dec vs 9B (size) |
|---|---|---|---|---|---|
| cuda factoid | 21.8 | 39.0 | 49.1 | **1.8×** | 2.3× |
| cuda multihop | 62.0 | 155.6 | 186.6 | 2.5× | 3.0× |
| mlx factoid | 14.6 | 37.8 | 55.2 | **2.6×** | 3.8× |
| mlx multihop | 36.3 | 128.1 | 208.2 | **3.5×** | 5.7× |

So decomposition delivers each correct answer for **1.8–3.5× fewer generated tokens than the same-size
monolith, and 2.3–5.7× fewer than the one-tier-larger 9B monolith — at equal answer quality.**

**But the token win *inverts* in energy — and that gap is the tool-capability result.**
Measured GPU energy per correct answer (radt `system/SMI - Power Draw`, GB10, integrated over the run,
judge-defined correct):
- **Structure (decomposed vs monolith_4b, both 4B): decomposed costs *more* energy on factoid** —
  **160.1 vs 123.6 J** (decomposed **+30%**), even though it spent **1.8× fewer decode tokens.** On
  multihop it is 11% lower (**435.1 vs 489.2 J**) because there the token gap is 2.5×. **A token-based
  cost proxy (FrugalGPT-style) would price decomposed at ~½; the joules say it is *more expensive* on
  factoid** — because top_k=5 makes its 3× document re-encoding (prefill, §F1) fully overwhelm the
  decode it saves. **On the M2 the inversion is far larger** — macmon total-SoC energy shows decomposed
  at **319 vs 156 J/correct (2.0×)** on factoid and **818 vs 607 J (1.35×)** on multihop vs the 4B
  monolith — the M2's 0.97 prefill fraction turns the token-cheapest arm into the energy-priciest. The
  metric everyone reports (tokens) and the metric that pays the power bill (joules) point in **opposite
  directions**, and only the prefill/decode split + the power listeners expose it — an API/token-boundary
  benchmark cannot. *(cuda J = GPU-only SMI; mlx J = macmon total-SoC — different bases, so read each
  within-device. This sharpened vs the weak-retriever run, where cuda structure energy was break-even.)*
- **Size (decomposed 4B vs monolith 9B): decomposition still wins on energy** — cuda factoid **160.1 vs
  198.3 J/correct (1.24×)**, multihop **435.1 vs 795.3 J (1.83×)**; mlx factoid **319 vs 320 J**
  (break-even — decomposition's prefill tax exactly cancels the 9B's per-token cost) and multihop **818
  vs 1358 J (1.66×)**. All at equal quality. The gap is smaller than the token gap for the same reason:
  decomposition's own prefill tax eats into it.

*(Caveats: cuda J = GPU-only SMI (CPU/retrieval excluded, but the LLM roles run on the GPU); mlx J =
macmon total-SoC — the two are different bases, comparable only within a device; judge run-to-run noise
≈±1–2/120 propagates ~1–2% into per-correct figures — small vs the effects.)*

*Choreo capabilities used: graph modularity (matched arms), per-stage token + prefill/decode tracing,
power listeners.*

---

## Finding 3 — The retry loop is a retrieval-refusal mechanism, not answer self-correction

### 1) What Choreo offers
The retry loop is **data-dependent control flow** — a cyclic edge firing only when a grader rejects an
attempt. Choreo executes and traces it **natively, per attempt**, recording for a single query the
sequence of attempts, which grader fired, and the answer at each. A per-request LLM server (Orca/vLLM)
sees each retry as an unrelated fresh request; a stock profiler cannot express same-query evolution
across attempts.

### 2) The bigger picture
The loop is marketed as self-correction — grade the answer, rewrite, try again. What does it *actually*
do across attempts, and is its eventual success predictable before the budget is spent?

### 3) The findings (e5 re-run)
- **It is dominantly a retrieval-refusal mechanism, not answer critique** *(verified from the router
  logs).* Of all retries, **86–100% fire on an empty `relevance=no` attempt** — factoid 100% cuda
  (35/35 monolith, 40/40 4b) / 86–100% mlx; multihop 92–97% cuda / 96% mlx. The small remainder are
  `relevance=yes, hallucination=yes` (0–7%) — a *non-empty* answer rejected on the hallucination check.
  So the loop is overwhelmingly re-refuse-and-re-retrieve, not "self-correction of reasoning." **This
  held at both retriever strengths** (MiniLM top-3 and e5 top-5) — the mechanism is structural, not an
  artifact of weak retrieval.
- **It repairs *retrieval*, not reasoning — and how well depends on retriever + model strength**
  *(verified, e5; retried queries joined to verdicts by question text)*. Of queries that retried, the
  fraction eventually correct is **cuda factoid 70–71%** vs **cuda multihop 8–14%**: on single-hop the
  strengthened retriever *succeeds on the reformulated query* (re-retrieval works), but on multi-hop the
  documents were fine and the *reasoning* failed, which re-retrieval can't fix. On the 4-bit M2 rescue
  is far lower (factoid 0–33%, multihop 5–9%) — the quantized model capitalizes on re-retrieval less.
- **So early-exit cost is task- *and* device-conditioned** *(verified, e5)*: dropping all retries costs
  **12.5–13.3 pp on cuda factoid** (the loop is genuinely valuable there — a strong retriever makes the
  refusal→re-retrieve→succeed cycle pay off), but only **3.3–5.8 pp on cuda multihop** and **0–3.3 pp on
  the M2**. A static attempt-0 cut is nearly free on multihop and on the 4-bit device but throws away
  real accuracy on cuda factoid — motivating a *dynamic* discriminator (candidate: answer stationarity
  across attempts) over a blanket cut. **This is the payoff of tracing the loop per-attempt: the retry
  budget's value is neither uniformly high nor uniformly low — it is a function of task × retriever ×
  quantization, which only per-attempt attribution surfaces.** *(Note: strengthening the retriever
  *raised* cuda-factoid rescue from ~22% under the old MiniLM run to ~70% — better retrieval makes the
  retry loop more, not less, worthwhile there.)*

*Choreo capabilities used: native data-dependent control-flow tracing, graph modularity, per-stage
tracing.*

---

## Synthesis — three signals a request→response harness reads, each corrected by Choreo

| Signal it sees | What it implies | What Choreo surfaces |
|---|---|---|
| output-token count | more tokens = more cost | prefill dominates — a 4-tok grader is 80–96% of the generator; 62% of prefill is redundant re-encode (F1) |
| model size | scale up to buy accuracy | quality is **retrieval-bound**: a 2.25×-larger 9B ties the 4B even when retrieval hits, and stays tied after the retriever is strengthened (§R). A 4B role-graph matches the 9B at 2.3–5.7× fewer tokens/correct and 1.2–1.8× less energy (F2) |
| token count = cost | fewer tokens ⇒ cheaper | vs the same-size monolith, decomposed spends ½ the tokens yet **+30% joules** on factoid — prefill re-encode more than eats the decode saving; tokens and joules point opposite ways, only phase-split + power shows it (F2) |
| the final answer | one answer per request | the retry loop is a retrieval-refusal mechanism (91–100%), repairing retrieval not reasoning (F3) |

**Through-line:** the internal cost and control structure of an agentic-RAG pipeline is exactly what a
request→response benchmark — including MLPerf's own `e2e-rag` at its API boundary — cannot see, and
Choreo's in-process, per-stage, per-attempt lens is what surfaces it.

**Open items:** (i) F3's *dynamic* early-exit discriminator (answer stationarity across attempts) is
proposed but not built/measured — the per-attempt data to do it is now joined (question-text match), so
this is buildable; (ii) F1's ~62% hoist saving is a *ceiling* Choreo prices, not an implemented
optimization (and "lossless" assumes position-canonical / PromptCache-style reuse under RoPE);
(iii) prior art to out-distance, not restate:
PromptCache / vLLM APC / RadixAttention (F1), FrugalGPT / cost-per-accuracy (F2), CRAG / Adaptive-RAG /
Huang-2023 (F3).

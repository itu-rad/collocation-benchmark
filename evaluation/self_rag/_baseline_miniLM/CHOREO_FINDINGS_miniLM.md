# Self-RAG — What Choreo Surfaces About Agentic-RAG Cost and Control Flow

*Choreo evaluation. All numbers below recomputed from the committed **quality** cells (the scored runs)
on 2026-07-22. Accuracy is scored three ways — exact-match and token-F1 (project scorer,
`_best_answer`) plus an **LLM judge** (Haiku, semantic factual equivalence; verdicts under
`evaluation/self_rag/judge/`) — because exact-match materially mis-ranks these arms (§0). Greedy
decoding → answers are byte-identical across runs, so R=1 quality is exact.*

## Setup — arms, models, and sizes

| Arm | Model | Structure |
|---|---|---|
| **monolith** | **Qwen3.5-9B** (bf16 on cuda / 9B-OptiQ-4bit on mlx) | one 9B model, **single call** (relevance + answer + hallucination in one JSON) |
| **monolith_4b** | **Qwen3.5-4B** | one 4B model, single call (same monolithic prompt) |
| **decomposed** | **Qwen3.5-4B**, one instance per role | 4B, **separate call per role** (relevance grader, generator, hallucination grader, query-rewrite) |
| **decomposed_shared** | **Qwen3.5-4B**, one instance + mutex | 4B, the same role calls through one shared instance |

- **Devices:** cuda = GB10/DGX Spark (bf16); mlx = M2 Pro (4-bit OptiQ). Same retriever/corpus
  (`rag-mini-wikipedia`), tasks factoid + multihop, N=120.
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
| **cuda factoid** | 0.450 / 0.516 / **0.725** | 0.608 / 0.679 / **0.750** | 0.650 / 0.709 / **0.750** |
| **cuda multihop** | 0.242 / 0.310 / **0.308** | 0.275 / 0.347 / **0.367** | 0.275 / 0.338 / **0.350** |
| **mlx factoid** | 0.467 / 0.514 / **0.683** | 0.575 / 0.634 / **0.692** | 0.575 / 0.653 / **0.733** |
| **mlx multihop** | 0.217 / 0.269 / **0.300** | 0.242 / 0.319 / **0.333** | 0.267 / 0.335 / **0.333** |

*(`decomposed_shared` omitted: byte-identical to `decomposed` under greedy.)*

**Under semantic judging the arms are quality-comparable — the EM ranking was a scoring artifact, not
a quality difference.** On cuda factoid the EM spread 0.450→0.650 **collapses to 0.725→0.750** (all
three within 2.5 pp). The apparent "the 9B is worst" was **exact-match penalizing verbose answers**:
the 9B answers a yes/no question with `"The Celsius crater on the Moon is named after him"` (correct,
EM=0) instead of the bare `"yes"`. Splitting cuda factoid by question type isolates it — on
**entity (wh-) questions the 9B is the *most* accurate arm** (entity EM 0.459 vs 0.426), its whole EM
deficit is on the yes/no half where it restates instead of parroting the literal token.

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

| | arm | p50 lat | p90 lat | mean lat | throughput (q/s) |
|---|---|---|---|---|---|
| **cuda factoid** | monolith 9B | 3.6 s | 9.2 s | 19.5 s* | 0.051 |
| | monolith_4b 4B | 1.7 s | 5.8 s | 2.4 s | 0.411 |
| | decomposed 4B | 1.5 s | **3.2 s** | 1.9 s | **0.513** |
| | decomposed_shared 4B | 1.5 s | 3.1 s | 1.9 s | 0.514 |
| **cuda multihop** | monolith 9B | 2.9 s | 10.6 s | 5.5 s | 0.181 |
| | monolith_4b 4B | 2.0 s | 7.5 s | 3.9 s | 0.254 |
| | decomposed 4B | 2.1 s | **4.5 s** | 2.8 s | **0.353** |

*\*9B mean is tail-heavy (mean ≫ p90): a few long-generating queries drag it — use p50 as the typical
latency.*

**Reads that frame the whole section:**
- **On GB10, at equal answer quality (all three ≈0.73–0.75 by the judge), `decomposed` wins on
  *performance*: lowest latency, 2× the throughput of the 9B, and the *tightest tail*** (p90 3.2 s vs
  5.8 s for monolith_4b, 9.2 s for the 9B). The tight tail is because decomposition replaces the
  monolith's variable-length generation with predictable short grader calls. (The 9B is not less
  accurate — see §0 — it is simply slower for the same quality.)
- **`decomposed_shared` matches `decomposed` on latency/throughput at 1/3 the model memory** (one 4B
  instance vs three).
- **Device dependence (mlx):** on the M2 the story flips on *speed* — `decomposed` is still more
  accurate (higher F1 / answered-rate) but **slower** than `monolith_4b` (mlx factoid p50 4.3 s vs
  2.5 s), because on the slow 4-bit device its 3× context re-encoding costs more than it saves. The
  quality gain persists; the speed advantage does not.

These baselines are what the deep dives below *explain*: **why** the 9B is worst (Finding 1/2), **why**
`decomposed` has the tightest tail and dominates on GB10 but not the M2 (Finding 1), and **why** the
token-cheapest arm isn't proportionally energy-cheapest (Finding 2).

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
  model size.** Prefill's share of LLM time (cuda): `monolith` (9B) **0.16**, `monolith_4b` (4B)
  **0.21** — both decode-bound — vs `decomposed` (4B) **0.54** and `decomposed_shared` **0.54**. The
  **fixed-size** step `monolith_4b (4B) 0.21 → decomposed (4B) 0.54** proves it is the *split*, not the
  9B→4B change. Multihop: 0.29 / 0.33 → 0.63. On mlx the profile is already prefill-heavy and
  decomposition deepens it: factoid 0.73 / 0.77 → **0.94**; multihop 0.84 / 0.87 → **0.97**. The
  industry's headline serving fix — decode-side continuous batching (Orca/vLLM) — targets the phase
  decomposition just shrank.
- **Output-token count barely predicts a role's cost, because prefill dominates.** Per call (cuda
  factoid decomposed): the relevance grader emits **4 tokens** and costs **394 ms**; the generator
  emits **5.8 tokens** and costs **475 ms** — so the grader is **83%** of the generator's per-call cost
  despite emitting ~30% fewer tokens (hallucination grader 80%). On mlx the graders reach **91–96%** of
  the generator per call. The cheap-output graders are expensive because each **re-encodes the full
  ~400-token retrieved context** (prefill), which dwarfs the 2–6 decoded tokens.
- **~62% of the decomposed prefill is the same documents re-encoded 3×.** The three doc-bearing roles
  account for **93% of all prefill** (108 s of 116 s on cuda; 506 s of 543 s on mlx) and each re-encodes
  the identical retrieved documents, so two of every three copies are pure recompute. **Sharing the
  model does not reclaim it:** `decomposed_shared`'s per-role prefill and prefill-fraction (0.54) are
  identical to `decomposed`'s — the mutex shares weights, not work. And the deployed prefix cache (vLLM
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
**Accuracy (semantic judge — see §0):** all arms are quality-comparable (cuda factoid 0.725 / 0.750 /
0.750 for 9B / monolith_4b / decomposed; the exact-match ranking was a formatting artifact). So "correct"
below is **judge-defined**, and the ledger is *not* distorted by exact-match's lexical penalty against
verbose answers — which matters, because that penalty fell hardest on the very arm (9B) whose
per-correct cost we are dividing.

**Decode tokens per correct answer** (robust — this is generation *length*, independent of the scorer):

| | decomposed(4B) | monolith_4b(4B) | monolith(9B) | dec vs 4B (structure) | dec vs 9B (size) |
|---|---|---|---|---|---|
| cuda factoid | 27.7 | 53.7 | 151.1 | **1.9×** | **5.5×** |
| cuda multihop | 66.5 | 150.2 | 212.8 | 2.3× | 3.2× |
| mlx factoid | 17.5 | 56.5 | 74.1 | **3.2×** | 4.2× |
| mlx multihop | 39.3 | 158.4 | 268.6 | **4.0×** | 6.8× |

So decomposition delivers each correct answer for **1.9–4.0× fewer generated tokens than the same-size
monolith, and 3.2–6.8× fewer than the one-tier-larger 9B monolith — at equal answer quality.** Against
the 9B this is a clean win: same quality, a fraction of the tokens, lower latency (§0).

**But the token win does *not* carry over to energy — and that gap is the tool-capability result.**
Measured GPU energy per correct answer (radt `system/SMI - Power Draw`, GB10, integrated over the run,
judge-defined correct):
- **Structure (decomposed vs monolith_4b, both 4B): break-even** — factoid **144.6 vs 137.8 J**
  (monolith_4b 5% *lower*), multihop **448.5 vs 457.2 J** (decomposed 2% lower). Both within
  measurement noise. Yet decomposed spent **1.9–2.3× fewer decode tokens.** A token-based cost proxy
  (FrugalGPT-style) would price decomposed at ~½; the joules say **identical** — because decomposition's
  prefill re-encoding (F1) burns exactly the compute its shorter decode saved. **The metric everyone
  reports (tokens) says 2×; the metric that pays the power bill (joules) says even. Only the
  prefill/decode split *plus* the power listeners expose the difference — and an API/token-boundary
  benchmark cannot.**
- **Size (decomposed 4B vs monolith 9B): decomposition wins on energy** — factoid **144.6 vs 259.0
  J/correct (1.8×)**, multihop **448.5 vs 889.4 J/correct (2.0×)**, at equal quality. (Under exact-match
  this looked like 2.5×; the fair judge-based gap is **1.8×** — EM had inflated it by under-counting the
  9B's correct answers. Reporting the honest number *requires* the semantic scorer.)

*(Caveats: SMI is GPU-only power (CPU/retrieval excluded, but the LLM roles run on the GPU); mlx energy
not yet pulled, so the joule comparison is cuda-only; judge run-to-run noise ≈±1–2/120 propagates ~1–2%
into per-correct figures — small vs the effects.)*

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

### 3) The findings
- **It is dominantly a retrieval-refusal mechanism, not answer critique** *(verified from the router
  logs).* Of all retries, **100% fire on an empty `relevance=no` attempt on factoid** (31/31 monolith,
  35/35 4b) and **~91% on multihop** (74/81 monolith, 79/81 4b); the small remainder (7/81, 2/81) are
  `relevance=yes, hallucination=yes` — a *non-empty* answer rejected on the hallucination check. So the
  loop is overwhelmingly re-refuse-and-re-retrieve, with only a ~9% multihop minority of true
  answer-critiques — not "self-correction of reasoning."
- **It repairs retrieval, not reasoning** *(verified)*: retries rescue **factoid ~22% vs multihop ~4%**
  of retried queries — it helps when attempt-0 fetched the wrong docs (common single-hop) and is
  near-futile when the docs are fine but the reasoning fails (multihop).
- **So a blanket early-exit's cost is task-conditioned** *(verified)*: dropping all retries costs
  **1.8pp on multihop** but **3.6–4.5pp on factoid** — which motivates a *dynamic* discriminator
  (candidate: answer stationarity across attempts) rather than a static attempt-0 cut. *(The
  stationarity cross-tab of ~96/97 doomed-when-stationary is a claim from a per-attempt reconstruction
  and is still pending an independent re-run — flagged.)*

*Choreo capabilities used: native data-dependent control-flow tracing, graph modularity, per-stage
tracing.*

---

## Synthesis — three signals a request→response harness reads, each corrected by Choreo

| Signal it sees | What it implies | What Choreo surfaces |
|---|---|---|
| output-token count | more tokens = more cost | prefill dominates — a 4-tok grader is 80–96% of the generator; 62% of prefill is redundant re-encode (F1) |
| model size | scale up to buy accuracy | a **4B role-graph matches the 9B single model at equal quality** for 3.2–6.8× fewer tokens/correct and 1.8–2.0× less energy (F2) |
| token count = cost | fewer tokens ⇒ cheaper | vs the same-size monolith, decomposed spends ½ the tokens but the **same joules** — prefill re-encode eats the decode saving; only phase-split + power shows it (F2) |
| the final answer | one answer per request | the retry loop is a retrieval-refusal mechanism (91–100%), repairing retrieval not reasoning (F3) |

**Through-line:** the internal cost and control structure of an agentic-RAG pipeline is exactly what a
request→response benchmark — including MLPerf's own `e2e-rag` at its API boundary — cannot see, and
Choreo's in-process, per-stage, per-attempt lens is what surfaces it.

**Open items:** (i) verify F3's stationarity cross-tab (the one number still from a subagent
reconstruction, not my own pass); (ii) F1's ~62% hoist saving is a *ceiling* Choreo prices, not an
implemented optimization (and "lossless" assumes position-canonical or PromptCache-style reuse under
RoPE); (iii) mlx energy (J/correct) not yet pulled — only cuda; (iv) prior art to out-distance, not
restate: PromptCache / vLLM APC / RadixAttention (F1), FrugalGPT / cost-per-accuracy (F2), CRAG /
Adaptive-RAG / Huang-2023 (F3).

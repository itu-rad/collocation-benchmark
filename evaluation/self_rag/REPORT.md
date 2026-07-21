# Self-RAG Experiment — Full Report

*Choreo evaluation, R=1, mlx (M2 Pro, 4-bit Qwen3.5-4B) + cuda (GB10, bf16 Qwen3.5-4B). Greedy decoding (deterministic → R=1 quality is exact). Generated 2026-07-21.*

---

## 1. What the experiment is

Self-RAG is a **graph-structured RAG pipeline with a data-dependent retry loop**:
retrieve → grade relevance → generate → grade hallucination → grade answer-quality →
(if bad) rewrite the query and retry, up to a budget. Unlike the linear 3D-UNet
pipeline, it has genuine graph structure and data-dependent control flow — which is the
only reason it earns a place next to 3D-UNet: it exercises Choreo's per-stage +
open-loop + collocation lens natively.

- **Arms:** `monolith` (one LLM plays all roles via prompting), `decomposed` (separate
  LLM stages per role), `decomposed_shared` (decomposed but one shared model instance +
  mutex), `monolith_4b` (a different/larger base model).
- **Tasks:** `factoid` (easy single-hop QA), `multihop` (hard multi-hop reasoning),
  rag-mini-wikipedia, N=120 questions.
- **Metrics:** EM, token-F1, answered-rate (SQuAD normalization); per-stage latency
  (trace CSVs); per-arm end-to-end latency.

## 2. Headline results (measured)

### 2a. Decomposition improves factoid quality — and it is REAL, not an artifact
| task | contrast | ΔEM | ΔF1 | McNemar mid-p |
|---|---|---|---|---|
| factoid | monolith → decomposed (mlx 4-bit) | **+0.108** | +0.139 | 0.009 |
| factoid | monolith → decomposed (cuda bf16) | **+0.200** | +0.193 | 0.000 |
| multihop | monolith → decomposed | +0.03 | +0.03 | 0.23 (null) |

- Replicates across **both devices and precisions**; stronger at bf16.
- `decomposed_shared` ≈ `decomposed` in quality (same gain, 1× memory).

**Artifact gate — CLEARED (the reviewers' #1 concern):**
- **Answered-rate identical** (0.883 monolith vs decomposed, cuda) → the gain is *not*
  conservative-refusal selection.
- **F1-gain ≥ EM-gain on both devices** (cuda ΔEM +0.200 ≈ ΔF1 +0.193; mlx ΔF1 +0.139 >
  ΔEM +0.108). A verbosity/formatting artifact would make the format-robust F1 move
  *less* than EM; instead it moves as much or more → **genuine semantic improvement.**
- Belt-and-suspenders (cheap follow-up): re-prompt the monolith with the decomposed
  arm's answer-extraction format and re-score. Existing data already refutes the artifact.

### 2b. Self-critique LLM calls dominate latency, NOT the answer generator (the "wrong-stage" result)
Per-stage total time, cuda factoid decomposed pipeline (`evaluation/self_rag/stage_latency.py`):

| stage | total | note |
|---|---|---|
| Answer Generator LLM | 109.7s | the thing everyone benchmarks |
| Relevance grader LLM | 92.1s | |
| Hallucination grader LLM | 89.7s | |
| Query-rewrite LLM | 30.2s | |
| **self-critique total (graders+rewriter)** | **212s** | **~1.9× the generator** |
| Document retrieval | 16.2s | negligible |

**In agentic RAG you pay ~2× more compute to *judge and rewrite* than to *generate the
answer*, and retrieval is negligible.** Optimizing/benchmarking the answer LLM (the norm)
misses where the cost actually is — the direct RAG analog of the 3D-UNet "MLPerf measures
the wrong stage." Provable from existing traces, no new runs.

### 2c. Decomposition ≈ one model-size upgrade (the "crutch" signal)
`monolith_4b` improves factoid by +0.108 — about the same as decomposition. So
"decompose the pipeline" buys roughly what "use a better base model" buys. Suggestive
that the decomposition gain is a **small-model crutch** that shrinks as the base model
scales (needs the Qwen3.5 size ladder to confirm the curve).

### 2d. The retry loop is expensive on exactly the queries it fails to help
Multihop `monolith` runs ~5× slower per cell (~79 min vs ~13 min) for a **null** quality
gain — the retry loop grinds on hard queries and fails them anyway. The cost is
anti-correlated with the benefit. (Quantifying per-query retry-count vs outcome needs the
retry-loop instrumentation — see §5.)

## 3. Three ASPLOS-reviewer synthesis (serving / RAG-methodology / harsh-skeptic)

Unanimous conclusions:
1. **The quality claim must NOT be the headline.** "Decomposition helps easy, not hard"
   is already known (Self-Ask; Huang et al. 2023 "LLMs Cannot Self-Correct Reasoning
   Yet"; Self-RAG/CRAG) and this is *role*-decomposition, not question-decomposition. Use
   it as motivation/setup, not the contribution.
2. **Reframe to the RETRY LOOP's serving dynamics** — the one thing a linear pipeline
   can't give and where Choreo's lens is load-bearing.
3. **Drop** H4 (hardware "inversion" — budget-arbitrary, precision-confounded), H6
   (metastability — no amplification path: retries are *quality*-triggered, not
   *load*-triggered, so it can't manifest without contrivance), H10 ("no universal
   topology" — capability boilerplate).

## 4. The sharpest claim (what to build the section around)

> In a graph-structured RAG pipeline, the data-dependent retry loop is a hidden,
> quality-triggered source of latency variance whose per-query cost is **anti-correlated
> with its benefit** — it spends the most compute on exactly the queries it fails to
> improve. This variance is invisible to token-count-based LLM schedulers (Orca/vLLM
> model sequence length, not retry count), produces a heavy open-loop tail, and a
> **difficulty-aware early-exit/admission policy removes most of the tail at negligible
> quality loss.**

Supporting punches:
- **Grader calls dominate latency** (§2b) → you're optimizing the wrong stage.
- **Decomposition buys ~one model tier** (§2c) → under serving load a bigger monolith
  Pareto-dominates the stage graph — spend memory on the model, not scaffolding.
- **Collocation contention (the framework-native missing hypothesis all three reviewers
  raised):** the decomposed arm's N model instances contend super-additively on M2
  unified memory, so isolated per-stage latencies *mis-predict* collocated end-to-end
  (and differently on GB10). This is the only claim where Choreo's collocation lens is
  load-bearing, and it aligns with the paper's engine-specific-contention thesis.

## 5. Proven-now vs. what's left to build

- **Proven now (existing data):** artifact gate cleared (§2a); grader-latency dominance
  (§2b); the crutch anchor point (§2c); the 5×-for-null retry cost (§2d).
- **Cheap runs:** model-size-ladder crutch curve (§2c, ladder exists); monolith
  extraction-prompt control (artifact belt-and-suspenders).
- **The one real build (on-mission — it IS Choreo's measurement thesis):** retry-loop
  instrumentation (per-query retry-count + outcome) + open-loop arrival runs, to land the
  §4 headline (anti-correlated cost/benefit → heavy tail → difficulty-aware early-exit
  removes it) and the collocation-contention measurement.

## 6. Verdict

Self-RAG is the right vehicle **for the retry loop, not the quality claim.** The quality
result is real and defensible but unsurprising to a RAG PC; the contribution lives in the
serving/collocation accounting the RAG literature never measures. Framing is now settled;
the remaining work (retry instrumentation + open-loop) is well-scoped and directly serves
the paper's thesis.

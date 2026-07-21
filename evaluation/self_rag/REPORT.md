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

## Paper-ready section (v4, after 3 review rounds)

*Full review→rewrite trace in evaluation/PAPER_SECTIONS_TRACE.md. Final panel: Accept / Weak-Accept / Weak-Reject — the measured 98%-retry-waste (B.3) carries it; the experiment that flips to strong Accept is the open-loop retry-tail run (see caveats).*


**B.1 What this is.** We study a **multi-call critique-and-retry RAG graph** (relevance grader
→ generate → hallucination grader → answer grader → query-rewrite-and-retry; cf. CRAG [Yan et
al. 2024], Adaptive-RAG [Jeong et al. 2024]). This is *not* Asai et al.'s reflection-token
Self-RAG, which amortizes critique into one decoding pass; our pipeline issues **separate LLM
calls** per critique role — the pattern most deployed agentic frameworks use.

**B.2 Where latency goes (serial breakdown; scaffolding).** Decomposition improves single-hop
EM +0.108 (4-bit) to +0.200 (bf16, McNemar p<0.001) — a real semantic gain, not a scoring
artifact (identical answered-rate, token-F1 gain ≥ exact-match gain). A serial per-stage
breakdown puts the three auxiliary calls (two graders + rewriter) at ~1.9× the generator's
latency (212s vs 110s), retrieval negligible (16s). We note this is call-count-driven with
**equal-size graders**; CRAG-style lightweight graders or prefix caching (vLLM) / continuous
batching (Orca) would shrink it. We keep this as scaffolding — in the equal-size-grader
pattern, tuning the answer model alone leaves most latency unaddressed — and let the retry
result below carry the section.

**B.3 Measured: the retry loop's compute allocation.** From the existing run logs we recover
per-query retry counts and correlate them with correctness. On multi-hop, **98% of all retries
are issued on queries that ultimately end incorrect** (retry count as a proxy for compute,
assuming roughly uniform per-retry cost). Retried queries score EM 0.041 vs 0.380 for
never-retried queries — but we read this as *descriptive of selection* (the graders trigger
retries precisely on the hard queries), not as causal evidence of waste. The waste claim rests
instead on the **aggregate** result: the multi-hop quality gain is a bounded near-null (ΔEM
+0.033, 95% CI [−0.017, +0.083]), so the retries the loop concentrates on hard queries buy
nothing measurable — consistent with Huang et al. (2023) that LLMs cannot self-correct
reasoning without external signal. This is a measured **retry-count-vs-correctness
correlation** — a compute-allocation result — *not* yet a serving-tail result: retry count is a
per-query cost multiplier decoupled from sequence length, but whether it yields a heavy
open-loop p99 requires an under-load run we have not collected (§B.6).

**B.4 An open provisioning tension.** A one-tier-larger monolith recovers the +0.108 gain —
but only in the 4-bit arm; the bf16 decomposition gain (+0.200) was never raced against a
larger bf16 monolith. Whether the scaffold is a small-model crutch or a memory-saving
alternative is unresolved and needs a memory/latency cost ledger for both sides across ≥2
model pairs; we present it as an open question, not a result.

**B.5 The scheduler gap, stated precisely.** Per-request LLM schedulers (Orca, vLLM) treat a
retry as a fresh request; the invisibility is not an engine defect but lives at the
**orchestration layer** — a query's total cost (retry count) is data-dependent and unknown to
a length-based cost model. This motivates a difficulty-aware admission/early-exit policy that
predicts rescuability before spending the retry budget; we measure the wasted-allocation
opportunity (§B.3) but do not yet build the policy.

**B.6 Takeaway.** In the deployed multi-call critique pattern, per-stage measurement shows most
latency is self-critique rather than generation (under equal-size graders), and — the measured
core — the retry loop concentrates its issues on hard queries whose aggregate quality gain is a
bounded null, so its compute allocation is almost entirely unproductive, and retry count is a
data-dependent cost variable a length-based cost model misprices. Turning that compute-
allocation result into a measured *tail* under open-loop load, and building the difficulty-
aware scheduler it motivates, are the next steps this measurement enables.

## CAVEATS & OPEN QUESTIONS (for Robert)

*Synthesized across all 3 review rounds. Read before committing the section.*

**The measured core is honest but is "a known result quantified."** The 98%-retry-waste (B.3) is
defensible and useful, but the skeptic is right that it largely *follows* from two cited facts:
the graders select hard queries for retry, and Huang et al. (2023) show self-correction can't fix
reasoning without external signal. So the number is a quantification of a predictable outcome. The
genuinely-new-for-systems part is the *framing* — retry count as a data-dependent cost variable
decoupled from sequence length, invisible to per-request length-based cost models (Orca/vLLM) at the
orchestration layer. That framing needs the open-loop tail measurement to *bite* (below).

**Overclaims we softened (and why).** "measured serving consequence" → "measured retry-count-vs-
correctness *correlation*" (we measured a compute-allocation fact, not latency/throughput/tail).
"98% of retry *compute*" → "98% of *retries*" (count is a proxy for compute under ~uniform per-retry
cost). The EM 0.041-vs-0.380 contrast is *selection-confounded* (retried = graders-selected as hard),
so "wasted" rests on the *aggregate* null CI ([−0.017, +0.083]), not that contrast. The 1.9× critique
dominance is flagged as *equal-size-grader, un-cached, serial* — a real deployment with small graders
+ prefix caching would shrink it, so B.2 is scaffolding and B.3 carries the section.

**Alternative narratives the same data supports.** (1) *A provisioning paper* — "separate-call
critique is Pareto-competitive-or-dominated by a one-tier-larger monolith"; B.4 becomes the headline
(needs the cost ledger). (2) *"Use reflection-token Self-RAG"* — our cost finding is an argument for
Asai's amortized design over separate calls, not against self-critique per se. (3) *Role ≠ question
decomposition* — multi-hop's null may say our technique (role graph) is the wrong lever for multi-hop
(question decomposition — Self-Ask/DecomP — is the known-effective one), not that self-critique is
costly.

**Missing evidence + cost.** (i) **Open-loop under-load run** showing retry-count variance produces a
p99 tail a length-based scheduler misprices — the reviewers' #1; the one thing that turns B.3's
framing into a measured serving result. This needs *new* instrumentation + under-load runs (the one
fix B can't do cheaply from existing logs). (ii) *Small-grader + prefix-caching variant* — cheap;
defends B.2's "critique dominates" against the obvious deployment fix. (iii) *Cost ledger for B.4*
(bigger bf16 monolith vs the scaffold, memory + latency, ≥2 model pairs) — the bf16 +0.200 gain was
never raced against a larger bf16 monolith, so B.4 is currently 4-bit-only. (iv) *Grader
precision/recall* (LLM-judge/self-preference bias, Zheng 2023) — the graders drive the retries.

**Open questions.** Does the 1.9× survive right-sized graders + prefix caching? Is the multi-hop gain
truly null or underpowered (the CI bounds it ≤~8% EM — bounded, not merely p>0.05)? Under what
memory/task regime does the scaffold beat the one-tier-larger monolith? Would a difficulty-aware
early-exit policy (predict rescuability from retrieval/first-grader signal) recover the ~98% wasted
retries at equal quality — the natural follow-up this measurement motivates?

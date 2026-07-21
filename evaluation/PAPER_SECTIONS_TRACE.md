# Paper-ready sections — 3-round review→rewrite trace

This document records the full trace the author requested: the draft paper-ready sections
(A = 3D-UNet/MLPerf, B = Self-RAG), then three rounds of ASPLOS-reviewer critique and
rewrite. Reviewer lenses each round: (1) systems/architecture, (2) benchmarking-methodology,
(3) harsh skeptic. The final version (v4) is inserted into the two experiment reports; the
caveats synthesized across rounds land in each report's "Caveats & Open Questions" section.

Status: Round 1 reviews in progress (skeptic in; systems + benchmarking pending). v2 rewrite
follows when all three land.

---

# v1 — DRAFT (pre-review)

## SECTION A — 3D-UNet / MLPerf: measurement blindness in a linear pipeline

**A.1 Motivation.** MLPerf Inference is the de-facto standard for reporting ML system
performance. By construction it isolates the model as the System-Under-Test: inputs are
preprocessed offline and preloaded into an in-memory Query Sample Library, and only the
model is timed, under SingleStream (closed-loop, one query at a time) or Offline
(saturation throughput). This isolation buys reproducibility, but it also fixes a
measurement boundary that may or may not reflect how the model is served. We ask the
narrowest possible version of the question — *does MLPerf's boundary capture real serving
cost even for a single, linear pipeline?* — using the MLPerf 3D-UNet/KiTS19 medical
segmentation benchmark, run head-to-head against MLPerf's own LoadGen harness on identical
hardware, model, and the 42-study inference set. We validate equivalence first: our
end-to-end run and MLPerf's harness produce the same segmentation accuracy (Dice 0.870 vs
the reference 0.8617), so any divergence below is in *what is measured*, not *how well*.

**A.2 The excluded work is hardware-dependent and grows as accelerators improve.** MLPerf
excludes the raw→model-ready preprocessing (resample/normalize/pad). Measured serially,
this is ~5% of end-to-end on an M2 Pro but **19% on a GB10 (up to 70% for the smallest
studies)** — because the GB10 runs inference 10× faster while the CPU preprocessing barely
changes. This is Amdahl's law applied to the measurement boundary: the faster the
accelerator, the larger the fraction MLPerf's inference-only number omits. We are careful
about scope: a pipelined execution overlaps preprocessing with inference (measured: serial
424s → pipelined 330s on GB10, 22% hidden), so for *throughput* MLPerf's offline model is
defensible. The exclusion bites on *per-request latency* and *serial* execution — and it
worsens on exactly the hardware practitioners are buying.

**A.3 The primary result: MLPerf is blind to a scheduling degree of freedom it permits.**
3D-UNet's sliding-window inference has 8–144 subvolumes per study depending on input
volume size, producing an **18–20× spread in per-study service time** (10–187s on M2,
1–20s on GB10). On the workload's real deployment — a single on-prem GPU doing nightly
batch cohort reprocessing plus modest, bursty live studies — this heterogeneity breaks
both MLPerf metrics. Because the subvolume count is known from the volume shape *before*
inference, a size-aware (shortest-job-first) schedule is realizable. Under it, routine
studies return their result **10.3× sooner on M2 and 11.2× on GB10** (M2: 28.3→2.7 min
mean time-to-result) — yet **MLPerf's Offline throughput is byte-identical for FIFO and
SJF**, because throughput is a function of makespan, which is order-insensitive. MLPerf
literally cannot distinguish a scheduler that returns routine results 10× sooner from one
that does not. Symmetrically, under modest open-loop load (ρ=0.35, far below saturation) a
routine study's p99 latency is inflated **9–10× by head-of-line blocking** behind large
studies — invisible to MLPerf's closed-loop SingleStream, which never queues. Both effects
reproduce with near-identical ratios across a slow edge device and a fast datacenter
accelerator, because they are governed by ordering and variance, not absolute speed. We
confirm the throughput-vs-flow-time divergence in MLPerf's *own* Offline harness: reordering
the issued batch shortest-first (which Offline explicitly permits) leaves reported
throughput unchanged while per-study flow times separate ~10×.

**A.4 Takeaway.** Even for a linear pipeline, MLPerf's order-insensitive throughput and
closed-loop latency are structurally blind to a scheduling degree of freedom that changes
routine-study time-to-result and tail latency by ~10× — a schedule the benchmark permits
but neither measures nor rewards. End-to-end, per-stage, open-loop measurement surfaces it,
and the preprocessing MLPerf hides *grows* as accelerators get faster.

---

## SECTION B — Self-RAG: the serving cost of agentic self-critique

**B.1 Motivation.** Agentic RAG pipelines wrap a generator in a self-critique loop —
grade retrieval relevance, grade the answer for hallucination and quality, and rewrite the
query and retry when a grader objects — to improve answer quality. This machinery is a
graph with data-dependent control flow, and its *serving* cost is essentially unmeasured:
the literature reports quality, not the end-to-end, per-stage, under-load cost of the loop.
We instrument a Self-RAG pipeline with a per-stage, collocation-aware framework across two
tasks (single-hop factoid, multi-hop) and two accelerators (M2 Pro 4-bit, GB10 bf16).

**B.2 The quality gain is real but is not where the cost lives.** Role-decomposition
improves single-hop EM by +0.11 (4-bit) to +0.20 (bf16, McNemar p<0.001) and is null on
multi-hop; the gain is a genuine semantic improvement, not a scoring artifact (answered-
rate is identical across arms and the token-F1 gain equals or exceeds the exact-match
gain, the opposite of a formatting artifact's signature). But a per-stage latency
breakdown shows the cost lives elsewhere: the **self-critique LLM calls (relevance and
hallucination graders plus the query rewriter) total ~1.9× the answer generator** (212s vs
110s), while retrieval is negligible (16s). In agentic RAG one pays roughly twice as much
compute to *judge and rewrite* as to *generate the answer* — so optimizing or benchmarking
the answer model, the near-universal practice, targets the wrong stage.

**B.3 The loop's cost is anti-correlated with its benefit.** On the hard multi-hop task the
retry loop runs ~5× longer for a *null* quality gain: it grinds on queries it cannot
rescue and fails them anyway. The mechanism that lifts quality on easy queries is pure
serving cost on hard ones, and because retries multiply per-query latency, it is a
data-dependent source of tail latency that token-count-based LLM schedulers (which model
sequence length, not retry count) cannot see. This motivates a difficulty-aware admission
/ early-exit policy that predicts rescuability before spending the retry budget.

**B.4 Provisioning implication.** A one-tier-larger monolith recovers the same single-hop
quality gain as the entire decomposition (+0.108), suggesting the self-critique scaffold is
a small-model crutch whose value should be weighed against simply spending the memory on a
larger model.

**B.5 Takeaway.** The quality-improving machinery of agentic RAG is a serving liability the
quality-only literature does not measure: its self-critique calls dominate latency, its
retry loop spends the most compute on the queries it helps least, and per-stage open-loop
measurement is what exposes both — and what a difficulty-aware scheduler needs to fix them.

---

# ROUND 1 — reviews (3 ASPLOS reviewers)

**Convergent verdict: both sections Weak Reject.** Cross-cutting: each leads with a *known*
systems fact (A: throughput≠flow-time; B: separate calls cost more) surfaced by the
framework — must instead lead with the genuinely-novel part and confront prior art early.

**Reviewer 1 (systems).** A Weak Reject: "invisible to MLPerf" ignores the **Server
scenario** (open-loop, p99-bounded) — reframe to "3D-UNet ships no Server task"; SJF/throughput
identity is a tautology, the real novelty is **service time predictable a priori** (SJF without
profiling) + cross-accelerator reproduction; preprocessing-grows is a CPU-frozen artifact
(DALI/pipelining dissolve it). B Weak Reject: mislabeled vs Asai Self-RAG (reflection tokens,
not separate calls — this is CRAG-like); 1.9× is orchestration-specific (prefix caching/Orca
collapse it); "under load" claimed but data is serial wall-clock; B.4 refutes the thesis
(bigger monolith dominates). Missing: run LoadGen **Server**; GPU-preprocessing variant; CIs;
disclose sim-vs-measured; monolith cost ledger.

**Reviewer 2 (benchmarking).** A Weak Reject (reframable): Server omission reads as concealment;
Offline order-insensitivity is deliberate/documented; 70% is a per-study max not representative;
disclose whether p99 is measured or simulated + service-time CV. Novel bits to lead with:
a-priori-knowable SJF key, unusually large spread, cross-accelerator reproduction. B Weak Accept:
serving-cost framing genuinely underexplored, but mislabeled (CRAG/Yan not Asai); B.4 is n=1,
single-hop, +0.108 only in 4-bit arm, no cost accounting; multi-hop "null" needs a CI/power
(you have the tooling); grader-reliability/LLM-judge-bias check (Zheng 2023).

**Reviewer 3 (skeptic).** A Weak Reject: headline is a category error/tautology (order-insensitive
metric insensitive to order); ρ=0.35 single-GPU-FIFO is a strawman deployment (separate queues
trivially fix HoL); Amdahl over-generalizes. B **Reject (current form)**: the retry-tail headline
**outruns the data** — needs retry-count instrumentation + open-loop runs NOT done; "5× longer"
is a mean not a tail; "spends most compute on queries it helps least" needs a per-query
correlation we lack; B.4 guts the thesis; "dominate" overclaims (1.9× = 63%, not >50%). Most
reject-worthy sentence: B.3's "data-dependent source of tail latency" (asserts a measured result
we cannot produce).

**Prior art all three expect cited:** Reddi et al. MLPerf (ISCA'20, all 4 scenarios); Schrage/SRPT
& Harchol-Balter (SJF optimality + starvation); Dean&Barroso Tail-at-Scale (CACM'13); Orca
(OSDI'22), vLLM/PagedAttention+prefix-caching (SOSP'23); Asai Self-RAG (ICLR'24), Yan CRAG (2024),
Huang et al. 2023 (self-correction); NVIDIA DALI.

---

# v2 — after Round 1

## SECTION A — Predictable-service-time scheduling that 3D-UNet's MLPerf scenarios do not reward

**A.1 Motivation and honest baseline.** MLPerf Inference isolates the model as the
System-Under-Test — inputs preprocessed offline and preloaded, only the model timed — under
one of four scenarios. For 3D-UNet/KiTS19 the suite ships **only Offline (throughput) and
SingleStream (closed-loop p90 latency)**; it defines a Server scenario (open-loop Poisson
arrivals under a p99 bound) that *would* expose queueing effects, but does not instantiate
it for this task. So the number practitioners cite for 3D-UNet is the Offline throughput,
which — by design and documentation — is makespan-based and order-insensitive: the SUT may
process the issued batch in any order, and reordering cannot change reported throughput.
That throughput ≠ flow time, and that shortest-job-first minimizes mean flow time, are
textbook (Schrage's SRPT). Our contribution is not that observation; it is what this
workload makes *realizable and measurable*, and what the shipped scenarios therefore fail
to reward.

**A.2 The novel lever: the optimal schedule is knowable a priori and the payoff is large
and hardware-invariant.** 3D-UNet's sliding-window inference issues 8–144 subvolumes per
study as a deterministic function of the input volume shape — so **per-study service time
(an 18–20× spread) is known before inference runs, from the header, without profiling or an
oracle.** Most models do not offer this. Consequently a size-aware (SJF) schedule is
directly implementable, and on the workload's deployment as a shared on-prem GPU it returns
routine studies **10.3× sooner on an M2 Pro and 11.2× on a GB10** (M2: 28.3→2.7 min mean
time-to-result), while the reported Offline throughput is byte-identical for FIFO and SJF.
The ~10× ratio reproduces across a 10×-faster accelerator because it is governed by the
a-priori service-time distribution, not by speed. The claim we defend is therefore narrow
and, we believe, novel: *for a workload whose service time is exactly predictable pre-
execution, the schedule that improves time-to-result an order of magnitude is realizable for
free, yet is invisible to the only performance number MLPerf ships for it.* (We verify the
throughput-vs-flow-time divergence directly in MLPerf's own Offline harness by reordering
the issued batch, which Offline permits; §A.4.)

**A.3 Head-of-line inflation, and its honest scope.** Under modest open-loop arrivals
(ρ=0.35, a trace-driven queueing analysis over the *measured* per-study service times, not a
hardware load test), a routine study's p99 is inflated ~9–10× by blocking behind large
studies — the expected consequence of high service-time variance under FIFO. MLPerf's
SingleStream (closed-loop) never queues and so does not surface it; MLPerf's Server scenario
would, but 3D-UNet ships none. We are explicit that (i) this is an M/G/1-style analysis over
real service times, not a live load test, and (ii) obvious operational fixes — separating
batch from live queues, or SJF admission — remove most of it; the point is that the shipped
3D-UNet metrics reward none of these choices.

**A.4 A secondary, weaker observation on the excluded preprocessing.** MLPerf runs
preprocessing offline. Measured serially, the excluded resample/normalize/pad is ~5% of
end-to-end on M2 but 19% on GB10 (a per-study maximum of ~70% on the smallest study, where
inference is near-zero) — because the GB10 accelerates inference 10× while the CPU
preprocessing does not. We stress this is a **CPU-resident, non-overlapped** figure: a
pipelined run hides 22% of it (serial 424s→pipelined 330s on GB10), and GPU-side
preprocessing (e.g., DALI) would close most of the rest. So for throughput MLPerf's offline
model is defensible; the exclusion bites only for strictly serial single-request latency. We
report this as a bounded caveat, not a headline.

**A.5 Takeaway.** For a workload with a-priori-predictable, highly variable service time,
the schedule that improves routine time-to-result ~10× (and tail latency ~10×) is free to
implement and reproduces across accelerators, yet none of the MLPerf scenarios shipped for
3D-UNet reward it — a concrete case for a latency-under-load scenario (or an end-to-end
metric) in the benchmark, which per-stage open-loop measurement supplies.

---

## SECTION B — The serving cost of multi-call critique-graph RAG

**B.1 Motivation, and what this is (and is not).** We study a **multi-call
critique-and-retry RAG graph** (relevance grader → generate → hallucination grader → answer
grader → query-rewrite-and-retry; cf. Self-RAG [Asai et al. 2024] and CRAG [Yan et al.
2024]). We note explicitly that this is *not* Asai's reflection-token Self-RAG, which
amortizes critique into a single decoding pass; our pipeline issues **separate LLM calls**
per critique role, which is the pattern most deployed agentic frameworks use and the one
whose serving cost is unmeasured. Our question is the serving cost of that machinery, per
stage.

**B.2 Where the latency goes (measured, serial).** On single-hop factoid, role
decomposition improves EM by +0.108 (4-bit) to +0.200 (bf16, McNemar p<0.001) — a real
semantic gain, not a scoring artifact (answered-rate is identical across arms and token-F1
gains equal or exceed exact-match gains). But a **serial per-stage latency breakdown** shows
the cost lives away from the answer: the three auxiliary LLM calls (two graders + rewriter)
sum to ~1.9× the generator's latency (212s vs 110s), with retrieval negligible (16s) for
this in-memory corpus. We are careful about what this does and does not show: it is
**latency, call-count-driven, in an un-cached, equal-size-grader, serial orchestration** —
prefix caching (vLLM), continuous batching (Orca), or right-sized (smaller) grader models
would reduce it, and web-scale retrieval would not be negligible. The defensible claim is
narrow: in the common deployed pattern, **the majority of pipeline latency is in critique,
not generation, so optimizing the answer model alone leaves most of it unaddressed.**

**B.3 The genuinely novel systems object: retry count is a control-flow tail-latency
variable.** On the hard multi-hop task the retry loop runs ~5× longer for a quality gain
that our data cannot distinguish from zero — consistent with Huang et al. (2023) that LLMs
cannot self-correct reasoning without external signal; our contribution is the *serving*
consequence, not the quality null. Because retries multiply a query's latency and are driven
by *data-dependent control flow*, retry count is a latency variable that sequence-length-
based LLM schedulers (Orca, vLLM) do not model. **We do not yet claim a measured tail:**
establishing the per-query retry-count distribution, the retry-vs-rescuability correlation
("compute concentrated on unrescuable queries"), and the open-loop p99 requires
instrumentation and under-load runs we have not yet collected. We frame these as the
motivated next step this per-stage measurement enables, and the target of a difficulty-aware
admission/early-exit policy — not as results in hand.

**B.4 An open provisioning tension.** A one-tier-larger monolith recovers the +0.108
single-hop gain (in the 4-bit arm; the bf16 decomposition gain of +0.200 is larger). Taken
at face value this argues the critique scaffold may be Pareto-competitive with — or
dominated by — simply spending the memory on a larger model. We flag this as an **open
question, not a result**: it rests on one model pair, one task, one precision arm, and lacks
a memory/latency cost ledger for both sides. Resolving it (cost both sides; ≥2 model pairs;
multi-hop) is required before either "scaffold is a crutch" or "scaffold saves memory" can
be claimed.

**B.5 Takeaway.** In the deployed multi-call critique pattern, per-stage measurement shows
the majority of latency is in self-critique rather than generation, and it exposes retry
count as a data-dependent tail-latency variable that length-based schedulers cannot see —
motivating (but not yet delivering) a difficulty-aware scheduler and an honest
scaffold-vs-larger-model provisioning study.

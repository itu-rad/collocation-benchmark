# Contention experiments — redesign proposal (E3′ + E6′)

**Status:** PROPOSAL (2026-07-12) — pending co-author sign-off (Rob + Ties + supervisor).
**Supersedes, if accepted:** E3 (VQA bandwidth contention 2×2) as designed in
`experimental_setup.tex` §E3, and the E6 foreground/background workload choice
(EfficientNet inference vs. fine-tune). The E6 *design* (B-sweep, per-process
attribution, degradation curve) is retained; only its workloads change.
**Convention:** follows `experimental_setup.tex` (Question / Design / Configurations /
Metrics / Threats / Collect-vs-report; no result numbers). Knobs follow the
pre-registered-rule protocol of `PAPER_TODO.md` §2.6.

---

## 0. Why redesign

### 0.1 What is wrong with E3 (VQA) as designed

Two fundamental problems, independent of collection quality:

1. **The diagnostic cannot attribute the effect.** E3's signal is "the heterogeneity
   advantage (CLIP-on-ANE vs CLIP-on-GPU) collapses from serial to pipelined,"
   attributed to DRAM-bandwidth contention. But *queueing dilution* predicts the
   identical observation with no bandwidth mechanism: CLIP is ~50–200 ms of work
   against a ~3 s LLM stage, so under pipelined load, per-query latency is dominated
   by queue wait at the LLM — identical in both mappings — and any delta in a
   non-bottleneck stage collapses arithmetically. The 2×2 has no observable that
   separates "bandwidth contention" from "the bottleneck stage dominates under load."

2. **The effect is mis-proportioned.** The bandwidth arithmetic: on the M2 Pro
   (~200 GB/s), Qwen3.5-9B 4-bit (5.6 GB weights) decoding at 20–30 tok/s streams
   ~110–170 GB/s — the LLM *alone* nearly saturates the bus. CLIP-L/14 (~0.6 GB) at a
   few encodes/s adds ~2–5 GB/s. The true contention effect of the co-located encoder
   is plausibly single-digit percent — below the noise floor of the planned cells, and
   smaller than the queueing artifact of (1). The experiment as proportioned would
   likely "confirm" the hypothesis for the wrong reason, and the planned
   `powermetrics` read would then show CLIP contributes almost nothing.

   Additionally, the design is not outcome-robust: if ANE-CLIP is *slower* than
   GPU-CLIP serially, there is no advantage to collapse and no cell tells a story.

### 0.2 What the slot must accomplish

E3's slot in the paper exists to (a) evidence the unified-memory bandwidth thesis —
the paper's one distinctive systems insight; (b) demonstrate a measurement that
single-engine, single-stream benchmarks structurally cannot make; (c) justify the
M2 Pro's heterogeneous engines being in the paper. It does **not** need to be
multimodal or a QA task — the VQA framing was incidental to the claim.

### 0.3 The replacement: ONE staged experiment (system → mechanism zoom)

**Decision update (2026-07-13, author's design):** E6′ and E3′ are presented as a
single staged experiment. The big-picture inter-pipeline result comes first, and
each subsequent stage *zooms toward the mechanism by changing exactly one thing in
the configuration* — so the section simultaneously delivers the collocation
evidence, the bandwidth law, and a live demonstration of the framework's
declarative reconfigurability (the paper shows the literal YAML diff at each
stage transition, the same "config is the experiment" move E7 makes):

- **Stage A — system view (= E6′ below):** RAG-serve foreground + B∈{0,1,2}
  indexing pipelines; degradation curve + per-process attribution. Planted
  observation: foreground loss tracks background *bandwidth*, not compute.
- **Stage B — isolate intensity:** B=1, sweep background rate.
  *Diff: loadgen block only.* X-axis becomes GB/s via the counters.
- **Stage C — purify the co-runner (= E3′ co-runners):** swap the indexer for
  single-resource co-runners (CPU stream / GPU encode / ANE encode if unblocked)
  at the same intensities. *Diff: background stage list.* Dose–response per engine.
- **Stage D — purify the foreground (= E3′ foreground):** swap RAG-serve for bare
  decode; split prefill/decode. *Diff: drop the retriever stage.* The negative
  control (compute-bound prefill flat, bandwidth-bound decode degrades) → the law.
- **Closing figure:** Stage-A points plotted ON the Stage-C/D dose–response curve —
  the system symptom sitting on the mechanism's prediction.

**Discipline rule:** every stage transition changes exactly ONE config element
(loadgen → background stages → foreground stages); a transition that changes two
things breaks the causal chain and is not allowed.

The cell matrices, pilot inputs, and knob rules of E3′/E6′ below are UNCHANGED —
the combination is presentational (Stage D's foreground doubles as E3′'s isolation
baseline, a mild collection saving). Tooling keeps the e3p_/e6p_ cell ids.
Target abstract sentence:

> On unified-memory systems from a 16 GB laptop SoC to a 128 GB Grace–Blackwell
> node, memory bandwidth — not compute — is the binding resource under collocation;
> \sysname{} measures and attributes it per stage and per pipeline.

### 0.4 Side effects of adopting this proposal

- The **VQA-accuracy scorer drops off the P0 list** (no OK-VQA quality claims remain).
  EM/F1 (E4) is still required.
- The CLIP/CoreML apparatus is **reused**, not discarded: the ANE encoder loop is an
  E3′ co-runner. The FAISS/Chroma + embedding stages are reused in E6′.
- Optional: keep ONE slim VQA cell (mapping A vs B under load, no 2×2, parity check
  only) as a half-page end-to-end illustration. Default: cut it.
- `experimental_setup.tex` §E3 and §E6 must be rewritten; the E6 B-sweep-generator
  requirement carries over unchanged (independent processes, own dataset stages).

---

## 1. E3′ — Unified-memory bandwidth interference (dose–response)

### Question

On a unified-memory system, every engine (GPU, ANE, CPU) draws on one physical
memory and its bandwidth. Is an "idle" engine actually free capacity — or does work
placed on it tax the bandwidth-bound stage running on the GPU? Quantitatively: by how
much does foreground LLM decode throughput degrade per unit of co-runner memory
traffic, and does the degradation depend on *which engine* generates the traffic?

### Hypothesis and predicted signatures (falsifiable, outcome-robust)

H1 (dose–response): foreground **decode** throughput (tok/s) degrades monotonically
with co-runner bandwidth demand (GB/s), approximately independent of the engine
generating the traffic.
H2 (phase split — the built-in negative control): foreground **prefill** (TTFT),
being compute-bound, degrades far less than decode under the same co-runner. This
distinguishes bandwidth contention from "the machine is busy" (thermal, scheduling),
which would hit both phases.
H3 (cross-device): the same signature appears on both DUTs, scaled by their bandwidth
headroom; on GB10 it additionally answers whether Grace-side CPU traffic taxes
Blackwell-side decode across NVLink-C2C — an open, timely question.

A **flat curve is also a publishable finding** ("co-runners are effectively free up to
X GB/s") — the design is interpretable under any outcome, unlike the VQA 2×2.

### Design

**Foreground:** a plain-generation pipeline (no retrieval, no retry loop — reuse
`pipeline_configs/generation_mlx.yml` / `generation_huggingface.yml` shapes):
fixed prompt set, greedy decoding, fixed generation length. Per query we record
TTFT (prefill) and per-token decode rate separately. Model: Qwen3.5-9B — 4-bit MLX
on M2 Pro, BF16 HF on GB10 (18 GB weights; at ~15 tok/s decode streams ~270 GB/s of
the 273 GB/s budget — i.e. *both* DUTs run the foreground near the bandwidth roof,
which is exactly where contention is measurable).

**Co-runners** — each an independent Choreo pipeline in its **own OS process** (this
is a collocation measurement; per-process listeners attach per pipeline), each driven
at a **fixed, rate-limited intensity** by the fixed-interval scheduler (the E5
MultiStream component — built once, used twice):

| Co-runner | Engine | Apparatus | Exists? |
|---|---|---|---|
| C1 GPU-compute | GPU (mps / cuda) | CLIP-L/14 encode loop, batch B_img | yes (`multimodal_vqa` stages) |
| C2 ANE | Neural Engine (M2 only) | Core ML CLIP encode loop | yes (`export_clip_coreml` + coreml stage) |
| C3 CPU-stream | CPU | memory-streaming stage: sequential read/write over a working set ≫ LLC (STREAM-like triad over a pre-allocated buffer) | **to build (~50 lines)** |
| C4 (optional) second decoder | GPU | a 2B/4B decode loop | yes (config-only) — connects E3′ to E4's decomposed regime |

**Intensity axis:** per co-runner, a pilot measures its saturated rate R_max
(encodes/s or sweeps/s); reported cells run at {0 (isolation), 25%, 50%, 75%, 100%}
of R_max. Co-runner bandwidth demand is estimated two ways and cross-checked:
(i) model-based — achieved rate × per-op traffic (weights + activations touched);
(ii) counter-based — the DRAM-counter delta vs. the isolation cell (§ Counters).

**Cells:** DUT × co-runner {C1, C2 (M2 only), C3} × intensity {4 levels} + one shared
isolation baseline per DUT ⇒ 13 cells (M2), 9 cells (GB10); C4 optional (+4).
Runs are minutes-scale ⇒ **R = 10** per the cheap-cell rule (`PAPER_TODO.md` §2.3).

### Counters (VERIFY FIRST — gate on this before committing the design)

The mechanism claim rests on a DRAM-bandwidth (or defensible proxy) read:

- **M2 Pro: ✅ RESOLVED — COUNTER-BACKED (verified 2026-07-13, M2 Pro / macOS 26.5).**
  `powermetrics` has **no** bandwidth sampler on this build (power/thermal rails only),
  and `macmon` exposes only `ram_power` (usable as a proxy). But the decisive channel
  exists: **IOReport "AMC Stats / Perf Counters"** exposes per-requestor DRAM **byte**
  counters — separate RD/WR (+ DCS RD/WR) channels for PCPU/ECPU, **GFX (GPU)**,
  **ANE**, AVD, DISP, etc. — readable **without root** at ≥1 Hz. Validation: idle
  ≈7 GB/s aggregate DCS; a ~20.5 GB/s induced CPU streaming load read back as
  ~27 GB/s CPU-attributed DCS (baseline + induced). This is better than the design
  assumed: not just total bandwidth but **per-engine attribution**, i.e. E3′ can
  report *who* moved the bytes. Probe: `scripts/ioreport_bw_probe.c`; runner:
  `scripts/preflight_bandwidth_counters.sh` (Probe 4).

  **Sampler built and validated (2026-07-13):** `scripts/amc_bandwidth_sampler.c`
  (compiled tool, streams CSV: per-interval CPU/GPU/ANE/other DCS rd+wr bytes +
  total GB/s; SIGTERM-safe, line-flushed) + `scripts/amc_bandwidth_sampler.py`
  (auto-compiles; CLI with `--label/--duration/--summary` writing
  `evaluation/results/<label>_bandwidth.csv`, and an `AMCBandwidthSampler`
  context manager for experiment drivers). Timestamps are wall-clock, matching
  the framework CSV's `%(created)f` for trace joins. Validation: CPU bucket
  1.8→24.7 GB/s under a ~19.5 GB/s induced CPU stream; GPU bucket 0.8→70 GB/s
  under an MPS matmul loop (total ~146 GB/s, near the ~200 GB/s roof).

  **Calibration notes (feed the E3′ pilot):** (i) sum only per-requestor
  DCS RD/WR channels — summing all AMC agents double-counts fabric hops
  (ATC/SB/AFI); (ii) the agent→engine mapping is not complete: under pure-GPU
  load ~half the traffic reports under AFR*/other agents, and kernel page-zero
  traffic from CPU allocations lands outside the PCPU/ECPU channels. Method:
  one single-engine calibration load per engine (CPU stream / GPU matmul / ANE
  encode) to empirically derive the mapping before collection — the pilot
  protocol already runs exactly these. (iii) ABI hazard, documented in both C
  files: `IOReportSimpleGetIntegerValue`'s second argument must be NULL —
  passing a local address corrupts the caller's stack (found the hard way).
- **GB10: ✅ RESOLVED — PROXY-BACKED (verified 2026-07-13 on babyxena/spark-cc0d).**
  `nvidia-smi utilization.memory` (memory-controller busy fraction) works and is
  the proxy of record; DCGM `PROF_DRAM_ACTIVE` is **confirmed unavailable** —
  nv-hostengine runs as a systemd service but its Profiling module reports
  "Failed to load" on this GB10 stack (checked 2026-07-14; not a privileges
  issue — closed); no usable
  Grace uncore events visible via unprivileged `perf` (the earlier "1 match" was
  a breakpoint pseudo-event, not a bandwidth counter). Consequence for wording:
  the bandwidth law is **counter-backed with per-engine attribution on the
  M2 Pro** and **proxy-backed (+ model-based GB/s estimates) on GB10** — state
  exactly that, no more.

If **neither** DUT yields a usable counter, the claim downgrades (as in the old E3)
to "consistent with bandwidth contention" — decide *before* collection whether that
weaker sentence still carries the section. The phase-split control (H2) partially
substitutes: a decode-only, prefill-flat degradation pattern is hard to explain by
anything but bandwidth.

### Knob table (rules per `PAPER_TODO.md` §2.6; values locked after pilots)

| Knob | Rule |
|---|---|
| Foreground prompt / gen length | fixed prompt set; 256 decode tokens; identical across all cells |
| Foreground queries per run | enough for ≥30 decode-rate samples post warm-up; pilot-derived |
| Warm-up | rolling-median flatness on pilot; ≥ first 3 queries + first co-runner minute discarded |
| Co-runner intensities | {0, 25, 50, 75, 100}% of pilot-measured R_max per (co-runner, DUT) |
| R | 10 (cheap cells) |
| Thermal | package power + clocks logged; pre-registered throttle threshold; throttled runs excluded |
| Placement/process | one process per pipeline; per-process listeners on |

### Metrics

Foreground: decode tok/s (primary), TTFT (control), per-query latency. Co-runner:
achieved rate, estimated + counter-measured bandwidth. System: DRAM counter/proxy,
package power, clocks. **Headline figure:** foreground decode tok/s (normalized to
isolation) vs. co-runner bandwidth (GB/s), one curve per co-runner engine, one panel
per DUT; TTFT overlaid as the flat control. **Headline statistic:** % decode loss per
GB/s of co-runner traffic (slope), with hierarchical run-level CI.

### Threats to validity

- *GPU co-runner (C1) confounds compute and bandwidth contention on the same engine*
  — by design: C1 is the "same-engine" reference; the clean off-engine evidence is C2
  (ANE) and C3 (CPU). State this; do not average across co-runner types.
- *Thermal coupling*: a co-runner heats the package; distinguished from bandwidth via
  H2 (thermal throttling slows prefill too; bandwidth contention should not) and via
  logged clocks.
- *Scheduler/runtime interference* (Metal command-buffer or CUDA context switching)
  for C1/C4 — again same-engine only; C2/C3 bypass it.
- *Model-based bandwidth estimates* assume weights-dominated traffic; state activation
  and KV-cache terms or bound them.
- Counter availability (§ Counters) is a pre-flight gate, not a post-hoc excuse.

### Collect vs. report

*Collect:* all cells above incl. C4 if cheap; both DUTs. *Report:* the two-panel
dose–response figure (decode + TTFT overlay), the slope table per (engine, DUT), and
one paragraph on the GB10 NVLink-C2C result (does CPU traffic tax GPU decode?).
C4 reported as a link to E4's decomposed regime if the signal is clean.

---

## 2. E6′ — Collocation scaling: RAG serving under index refresh

### Question

(Unchanged from E6:) as independent background workloads are added on shared
hardware, how does a latency-sensitive foreground degrade, and can \sysname{}
attribute the interference per pipeline? Re-instantiated with the workload pair every
production RAG deployment actually runs: **online QA serving collocated with corpus
(re-)indexing** — a background that is genuinely bandwidth- and compute-heavy
(properly proportioned, unlike an encoder against a 3 s LLM), and immediately legible
to reviewers.

### Design

**Foreground (fixed across all cells):** a *plain* RAG serving pipeline — Chroma
retrieve top-3 → Qwen3.5-4B generate (64–128 tok answers), driven by Poisson arrivals.
Deliberately **not** Self-RAG (no retry loop noise; service time ~1–4 s keeps the
≥500-pooled-queries p95 gate affordable: ~100 queries/run × R=5 ≈ minutes-scale runs).
Foreground rate λ_fg: pilot-derived so the B=0 cell sits at ~30–50% of capacity
(headroom for degradation to show; not a strawman), then **held fixed across all B**.

**Background (the B axis):** B ∈ {0, 1, 2} independent **indexing pipelines**, each:
document source → chunk → embed (GPU-placed embedder, e.g. MiniLM/bge-small) →
insert into ChromaDB. Each background is a **separate OS process with its own dataset
stage and its own target collection** (attribution requirement — carried over from
the original E6; `torchvision_mixed.yml`'s shared `dataset_stage_id: 0` must not be
reused). Background runs saturating (Offline scheduler) within each cell.

**Critical control — separate stores:** backgrounds index into collections/DB
instances the foreground never reads. Otherwise the foreground's retrieved documents
(and thus its answers and its service times) change as the index grows, confounding
*resource* interference with *behavioral* change. With separate stores, the only
coupling is hardware.

**Secondary axis (intensity, at B=1):** sweep the background's embed rate
{25, 50, 100}% of saturated via the fixed-interval scheduler — separates "a co-runner
exists" from "how hard it pushes," and directly overlays onto E3′'s dose–response
axis (same x-units: GB/s).

**Corpus:** reuse the FlashRAG/HotpotQA 20k-passage corpus (already an E4 artifact)
or wiki chunks; fixed chunking; each background gets a disjoint shard so B pipelines
do identical-shaped, independent work. **Bound the run window** (fixed number of
documents per cell) so insert cost doesn't drift with index size within a cell.

### Configurations

- New stages (small): `EmbedStage` (sentence-transformers, device-placed) and
  `ChromaIndexer` (insert-only). Chunker can live in the dataloader.
- The **B-sweep config generator** (already a PAPER_TODO §2.2 item) emits: foreground
  config + B background configs with disjoint shards, own dataset stages, own
  collections.
- Cells: DUT × B ∈ {0,1,2} × R=5, plus the B=1 intensity sub-sweep × R=5.
  Both DUTs; the GB10 (128 GB) vs M2 (16 GB) B-ceiling contrast is itself a
  portability data point. Report the ceiling reached; never extrapolate.

### Knob table (rules)

| Knob | Rule |
|---|---|
| λ_fg | pilot: B=0 at 30–50% capacity; fixed across all B |
| Foreground queries | ≥500 pooled per cell (p95 gate): ~100/run × R=5 |
| Background work unit | fixed docs-per-cell window; disjoint shards per background |
| Embedder + placement | one embedder, GPU-placed, identical across cells; stated |
| queue_depth | never-blocks rule (arrival trace verified) |
| Warm-up | foreground: rule-derived k; background: first minute (model load + first inserts) excluded from the reported window |
| R | 5 (cells are heavier); raw run values printed |

### Metrics

Foreground **p95 + median latency and throughput vs. B** (the degradation curve; p95
clears the ≥500 gate by construction), background indexing throughput (docs/s and
embeds/s), **per-pipeline attribution**: GB10 — `nvidia-smi pmon` per-process SM
activity, reconciled against the aggregate device counter (faithfulness check);
M2 — residency-only (stated limitation). Plus the E3′ DRAM counter/proxy per cell,
which ties the observed degradation back to the mechanism experiment: the E6′
foreground losses should sit *on* (or near) the E3′ dose–response curve for the
corresponding measured background bandwidth. That reconciliation — consequence
matching mechanism across two independent experiments — is the strongest single
piece of evidence the contention story can have.

### Threats to validity

- Background non-stationarity (index growth) — bounded by the fixed docs-per-cell
  window and disjoint shards.
- Foreground behavioral drift — excluded by the separate-stores control (verify:
  foreground answers byte-identical across B under greedy decoding; this doubles as
  the determinism check).
- Attribution is causal only where a per-process counter exists (GB10); M2 is
  residency-only — stated, as in the original E6.
- Embedder choice is a workload parameter, not a finding — one embedder, provenance
  stated, held fixed.
- Memory-capacity ceiling on the 16 GB M2 bounds B — reported as the ceiling, which
  is itself the portability point.

### Collect vs. report

*Collect:* B-sweep + intensity sub-sweep, both DUTs. *Report:* the foreground
p95-vs-B degradation curve with per-process attribution (the headline collocation
figure), the E3′-reconciliation overlay, and the per-DUT ceiling. Intensity sub-sweep
if space allows.

---

## 3. Decision checklist & sequencing

1. [~] **Pre-flight counter verification on both DUTs** (E3′ § Counters) —
   **M2 Pro: DONE, counter-backed** via IOReport AMC per-requestor byte counters
   (no root; per-engine attribution — see § Counters). **GB10: pending** — Ties runs
   `scripts/preflight_bandwidth_counters.sh` (Linux branch probes nvidia-smi
   `utilization.memory`, DCGM `PROF_DRAM_ACTIVE` field 1005, and Grace `perf`
   uncore events).
2. [ ] Co-author sign-off (Ties + supervisor) on: replacing E3-VQA with E3′;
   replacing E6's workload pair with E6′; VQA cut vs. kept-as-illustration.
3. [ ] Build delta (mostly shared with existing PAPER_TODO items): fixed-interval
   scheduler (already E5), B-sweep generator (already §2.2), + new: C3
   memory-streaming stage (~50 lines), `EmbedStage` + `ChromaIndexer`.
4. [ ] Pilots (§2.6 protocol) → lock knob tables → fold both experiments into the
   third-edition setup text (E3′ and E6′ sections replace E3/E6) and into Ties's
   GB10 collection spec.
5. [ ] Update `experimental_setup.tex` reconciliation notes: E3′ absorbs the old E3's
   "required DRAM read"; E6′ inherits E6's attribution machinery and scope-honesty
   rules verbatim.

**Cost estimate:** E3′ ≈ 22 cells × R=10 × minutes ⇒ roughly a day per DUT including
pilots. E6′ ≈ 6 cells + 3 sub-sweep cells × R=5 × ~10–20 min ⇒ 1–2 days per DUT.
Build work ≈ 2–4 dev-days, most of it already on the TODO for E5/E6.

# Path to a Publishable Paper — Master Checklist

Target: a submission-ready systems paper on **\sysname{} (Choreo)** —
a collocation-aware, end-to-end, graph-structured ML-pipeline benchmarking
framework for unified-memory devices (Apple M2 Pro + NVIDIA DGX Spark GB10).

This checklist is the single source of truth for what remains between "well-designed
plan" (where we are) and "publishable" (where we're going). It folds in the three
ASPLOS-style review rounds on `experimental_setup.tex` (all ended at *Accept on design,
conditional on collection*) **and the full repo/code/data audit of 2026-07-12**
(advisor review: framework code, evaluation artifacts, statistics, and paper docs).

**Legend**
- `[ ]` todo · `[~]` in progress · `[x]` done · `[?]` needs verification of status
- **P0** = on the critical path (a headline claim is unreportable without it) ·
  **P1** = needed for a complete paper · **P2** = strengthens / polish
- **[M2]** runs on the Apple M2 Pro · **[GB10]** needs the DGX Spark (Ties has access
  and will run collection) · **[dev]** local tooling, device-independent

---

## 0. Critical path (do these in order)

Re-ordered after the 2026-07-12 audit. The old spine ("GB10 data exists, just analyze
it") is **superseded**: the headline Self-RAG data is suspect until the `query_id`
provenance is resolved (§1.5), and the working assumption is now a **full GB10
re-collection, run by Ties**. Tooling, statistics, framework fixes, and the
hyperparameter protocol must all land *before* that re-collection so we only collect once.

1. **[x] P0 DATA INTEGRITY — RESOLVED (2026-07-13): pre-fix → full E4 re-collection**
   confirmed by the author. All headline Self-RAG data (GB10 + M2) is superseded;
   the write-ups must not be quoted. Ties runs the GB10 pass.
2. **[x] P0 [dev] EM/F1 scorer BUILT + prompt decision EXECUTED (2026-07-13)** —
   `evaluation/scripts/score_quality.py` (+ 14 unit tests). Initial re-score found
   EM = 0.000 (long-form answers vs short goldens); fixed by the **shared short-answer
   style** (`stages/self_rag/prompt_style.py`, identical in every arm) and validated
   end-to-end: EM 0.600 / F1 0.703 both arms, no answered-rate collapse. See §2.1.
   Small leftover: sidecar stores decomposed answers as stringified lists (§2.1).
   VQA-accuracy scorer deferred pending the E3′ redesign decision.
3. **[x] P0 [dev] Statistics migration DONE (2026-07-13)** — hierarchical bootstrap +
   run-level values + paired E2 statistic implemented in both overhead libs and all 7
   analyzers; tables regenerated. **Finding: the E2 mps tracing-off arm is
   contaminated** (per-run medians 120.3/96.5/90.2/90.3/89.4 ms — pooling hid it);
   old "+0.55% [+0.41,+0.69]" is superseded by "indistinguishable from zero" with two
   outlier runs → **re-collect E2 mps** (§3.2). Tracing-on is clean: +2.0% significant.
   E3/E4 analyzer CIs still to add when those experiments' final form lands.
4. **[x] P0 [dev] Framework fixes that gate final collection** (§2.5) — **DONE, working
   tree (uncommitted)**: polling busy-wait → blocking Condition wait, inference-path
   `zero_grad` guarded, `IndexRouter` typo, coordinated-omission telemetry, hygiene.
   Polling fix verified **end-to-end** (factoid_decomposed_mlx serial: 10/10, 0 spam,
   distinct query_ids); others unit-verified against real code paths. Router-dict cleanup
   deferred (§2.5). Nothing is re-collected on a build that still has these.
5. **[~] P0 [dev/M2] Hyperparameter protocol — M2 HALF DONE (2026-07-13).**
   `evaluation/pilots/` package built + idle-session pilots run (12 cells) →
   `knobs.yml` derived (per-task E4 λ: factoid 0.117, multihop 0.0217 q/s;
   E5 53.6 q/s; E6 fg 0.55 q/s; E1 warm-up k=22 — not 1; E2 k=200 confirmed) →
   configs locked via apply_knobs → `knob_tables.tex` (6 tables) generated.
   **Remaining:** GB10 pilots by Ties (`evaluation/pilots/README.md` GB10 spec);
   ANE/CoreML pilot cells hang (E3 mapping-B apparatus broken — investigate,
   §3.3); E5 pilot now runs on Imagenette (§3.5 dataset decision).
6. **[x] P0 E1 re-collection VERIFIED + warm-up corrected (2026-07-13)** —
   committed matrix is idle-M2 data (`run_matrix_env.txt`); pilot detector set
   warm-up k=22 (was 1-of-101); `table1.tex` regenerated — depth-flatness
   robust to the change (<1.5% shift). Remaining E1 item: synchronous tracing
   cost (§3.1).
7. **[~] P0 [dev] Build the delta of missing arms** (§2.2) — remaining after the
   2026-07-13 sweep (saturating-Offline scheduler, 4 multi-hop controls, and E7
   size rungs are DONE): **fixed-interval MultiStream scheduler** (E5);
   the **staged-contention apparatus** — C3 CPU memory-streaming stage,
   EmbedStage + ChromaIndexer, and the Stage-A–D config generator (single-diff
   transitions, redesign doc §0.3); **ANE-hang diagnosis (timeboxed)** — Stage C
   ships with CPU+GPU co-runners if unfixed. The N-in-flight scheduler is
   **DROPPED** (its only consumer was the cut VQA contended cells).
8. **[ ] P0 [GB10, Ties] Full GB10 collection pass** — E4 matrix (post-fix, locked knobs),
   E2 cuda half (cheap to re-run rather than argue provenance), E5 scenarios, E6 minimal
   sweep, E7 rungs. One pass, one protocol, one env capture.
9. **[ ] P0 [M2] M2 collection delta** — E3 2×2 (+ powermetrics), E4 timing 2×2 +
   multi-hop + quality runs at matched load, size rungs.
10. **[ ] P0 [dev] Run all analysis scripts → final tables & figures** (§4); regenerate
    the stale result `.md`s (§4.1).
11. **[ ] P0 [dev] Write the Results section** (§5.2) — **findings-first ordering**:
    E3/E4/E6 lead, E1/E2 as the short "trust the instrument" pass, E5 as positioning.
12. **[ ] P1 Third-edition setup text** (merge methodology.tex + experimental_setup.tex),
    incl. the knob tables (§2.6); resolve every `[[FILL]]`; unify naming (§5.1, §6).
13. **[ ] P1 Complete Intro / Background / Framework / Conclusion / Abstract** (§5) —
    honest framing of zero-copy and collocation claims (§5.3).
14. **[ ] P1 Artifact + reproducibility pass** (§7) — incl. the audit's landmine list
    (§7.1) — then **submit** (§8).

**Definition of "publishable":** every claim in the Results section is backed by
collected data on the relevant device(s), from post-`query_id`-fix runs, measured with
the metric the Methodology promises (EM/F1, not containment), with hierarchical
run-level CIs and raw R-run values shown; every config knob traces to a pre-registered
rule; no `[[FILL]]` remains; the framework name is consistent; methodology.tex and
experimental_setup.tex do not contradict; the artifact reproduces the reported numbers
from released traces (which must actually be released — see §7).

---

## 1. Author decisions — RESOLVED (updated 2026-07-12)

- [x] **Framework name → Choreo.** Keep `\sysname{}=Choreo`, sweep prose
  McBenchface→Choreo, fix methodology Table 2. → §6.2
- [x] **Setup text → a "third edition" (final)** merging `methodology.tex` (richer
  DUT/metrics prose) + `experimental_setup.tex` (7-experiment structure,
  collect-vs-report). Produce it → §5.1.
- [x] **Scope → E1–E5 non-negotiable. E7 merges into E4.**
- [x] **E6 → NON-NEGOTIABLE, minimal form** (decision upgraded from "negotiable" per
  advisor review, author agreed 2026-07-12). The paper is titled and framed around
  collocation; without E6 it is a framework paper with its headline capability asserted,
  which is a rejection pattern. Minimal form: GB10, one fixed foreground,
  B∈{0,1,2}, per-process SM attribution; M2 residency-only curve optional. → §3.6
- [x] **GB10 data → SUPERSEDED.** The old decision ("already collected, obtain + analyze")
  no longer stands: provenance vs the `query_id` fix is unresolved (§1.5), the raw
  CSVs/JSONL behind the published write-ups are not in the repo, and the committed
  Self-RAG CSVs lack the perf-counter column. **Working plan: full GB10 re-collection,
  run by Ties**, after tooling/fixes/knobs land (§0 items 2–7).
- [x] **Venue → ASPLOS as stretch target, with an explicit ladder:** MLSys (best fit for
  framework+methodology), EuroSys/ATC, ISPASS/IISWC (measurement-focused, receptive).
  Write for ASPLOS first; the paper only gets stronger for the others. Deadline choice
  still open — depends on GB10 re-collection speed.
- [x] **E7 ceiling → model-size sweep folded into Self-RAG (E4);** 27B ceiling rung
  optional (§3.7).

---

## 1.5 Data integrity — the `query_id` provenance gate — **P0, FIRST**

The bug: `Query.query_id` was a dataclass default evaluated once at import
(`utils/schemas/query.py`), so **all queries in a run shared one UUID** until fix
commit `b1e88eb`. `BinaryRouter`/`MonolithRouter` key their retry budgets on
`query_id`, so pre-fix, `max_retries: 2` meant **2 retries total per run, not per
query** — after exhaustion, every later failing query skipped its rewrite loop and went
straight to the error end-state.

Why this can invalidate the flagship result: the headline multi-hop numbers — monolith
"bails on 19/30" (GB10) and "answers 2/10" (M2) — are exactly the signature a global
retry budget produces, and the artifact hits the two arms asymmetrically (one long
monolith pass vs many short decomposed calls burn the shared budget differently).

- [ ] **Determine collection date/commit of the data behind
  `monolith_vs_decomposed_dgxspark.md` and `_m2pro.md`** relative to `b1e88eb`.
- [ ] Direct test where raw files exist: do the `_outputs.jsonl` show **distinct
  `query_id`s within a run**? (Pre-fix runs show one constant id per run.)
- [ ] If pre-fix (working assumption): mark both write-ups "superseded — do not quote,"
  and fold the E4 matrix into the GB10/M2 re-collection (§0 items 8–9).
- [ ] If post-fix: recover the raw CSVs (+ perf column) and JSONL into the repo/data
  release so the numbers are backed (§7).
- [ ] Either way: add a collection-protocol rule — every result doc records the git
  commit, config hash, and env capture of the run that produced it.

---

## 2. Tooling & analysis prerequisites (gate collection — build first)

### 2.1 Quality scorers — **P0 [dev]** (blocks E3, E4 — and may change verdicts)
- [x] **EM + token-F1 scorer built (2026-07-13):** `evaluation/scripts/score_quality.py`
  — SQuAD-style normalization, Wilson 95% + run-level cluster-bootstrap intervals,
  multi-run arms via globs, containment kept as secondary next to mean answer length,
  parity; 14 unit tests in `test_score_quality.py`. Writes `quality_report.{md,json}`.
- [ ] Implement **VQA accuracy** (soft match over the OK-VQA annotator answer set) —
  **deferred**: drops entirely if the E3′ redesign is accepted.
- [x] **Re-scored the committed smoke outputs.** Finding (2026-07-13): **EM = 0.000 on
  BOTH arms; F1 = 0.156 (monolith) vs 0.189 (decomposed)**; containment said 0.8/0.7.
  Long-form sentence answers vs short goldens ("yes", "sixteenth") make EM
  unattainable and dilute F1.
- [x] **DECISION RESOLVED + EXECUTED (2026-07-13): short-answer prompts adopted.**
  The strict short-span answer style lives in **one shared constant**
  (`stages/self_rag/prompt_style.py:SHORT_ANSWER_STYLE`) imported by both
  answer-producing formatters (`monolith_formatter.py` answer field,
  `answer_generator_formatter.py`), so arms can never drift — cite the shared
  constant in the knob table (§2.6) as the "identical across arms" guarantee.
  **Validated end-to-end** (factoid MLX, 10 q/arm, post-fix build):
  EM 0.000→**0.600**, F1 0.16/0.19→**0.703/0.703**, answer length 207/167→9/8
  chars; graders accept short answers (answered 10/10 monolith, 8/10 decomposed —
  the misses are the known retry-exhaustion class, not the new style). Remaining
  EM misses are genuine signal (e.g. entity answered where golden is 'yes').
  ⚠ Invalidates comparability with ALL pre-change quality data — already superseded
  by the §1.5 re-collection decision, so no additional loss.
- [ ] **Sidecar hygiene (small, before re-collection):** the decomposed arm writes
  `generated_answer` as a stringified list (`"['yes']"`) — normalization absorbs it
  in scoring, but store a plain string in the JSONL (find where the router/capture
  sets `generated_answer`) so the released artifact is clean.
- [ ] Keep containment as a *secondary* signal only, always reported next to **mean
  answer length** (gameable by verbosity — the arms measurably differ: 207 vs 167 chars
  in the smoke set).
- [ ] Emit **Wilson 95% intervals** on the rate metrics, plus a **run-level interval**
  over the per-run rates.
- [ ] Unit-test the scorers on a tiny fixture (known EM/F1/VQA values).

### 2.2 Missing load generators & config generators — **P0 [dev]** (unconditional slice DONE 2026-07-13)
- [x] **Saturating-Offline scheduler** — `loadgen/schedulers/saturating_offline_scheduler.py`
  (`SaturatingOfflineScheduler`, registered in `loadgen/__init__.py`). All samples enqueued
  up front, no pacing, no completion wait → MLPerf **Offline** throughput. Unit-tested
  (50 queries clustered in 0.4 ms, event set, terminator sent). Naming hazard documented
  (the old `OfflineLoadScheduler` is actually SingleStream).
- [x] **~~Fixed-interval MultiStream scheduler~~ — DROPPED** (decision 2026-07-13). E5 does
  NOT need to replicate all of MLPerf: the reduction is structural and the substantive
  claim (below) targets the *Offline throughput* number. MultiStream is a niche
  (multi-camera) scenario that maps to none of our workloads (ResNet/RAG/VQA) and is the
  fiddliest to get faithful. In the E5 table it is marked "expressible (constant-offset
  special case of the Poisson generator)" without an instantiated arm.
- [ ] **N-in-flight closed-loop scheduler** — **still pending, contingent on the E3′/E6′
  redesign** (`CONTENTION_EXPERIMENTS_REDESIGN.md`): it's the recommended E3 stationary
  contended operating point, but E3′ may replace the VQA 2×2 with a bandwidth
  dose–response experiment. Build once that decision lands.
- [ ] **E6 B-sweep config generator — still pending, contingent on E6′** (the redesign may
  swap the EfficientNet foreground/background for RAG-serving; the generator *mechanism* is
  workload-agnostic but the configs it wires depend on the call). Independent bg processes,
  each own dataset stage (torchvision_mixed.yml shares `dataset_stage_id: 0`).
- [x] **E7 size-ladder configs** — `factoid_monolith_{0.8b,2b,27b}_{mlx,cuda}.yml` created
  (6 files; 4B/9B already exist). Model-field-only swaps off `factoid_monolith_*`
  (verified diff = name label + 2 model fields → the "no-code" artifact). 27B is the M2 Pro
  OOM ceiling rung; BF16 cuda counterparts for GB10. All parse against `BenchmarkModel`.
- [x] **4 multi-hop Self-RAG control configs** —
  `multihop_{monolith_4b,decomposed_shared}_{cuda,mlx}.yml` created. monolith_4b = 9B→4B
  swap; decomposed_shared = shared transform (`depends_on_id:3` on stages 6 & 8,
  `tokenizer_stage_id` 6→3 & 8→3). Retrieval held fixed (reuse base `collection_name`).
  All parse against `BenchmarkModel` (8 / 13 stages).

### 2.3 Statistics harness — **P0 [dev]** (touches every experiment)
- [x] **Hierarchical (cluster) bootstrap implemented (2026-07-13)** in `noop_lib.py`
  and `modularity_lib.py` (resample runs first, then queries; 10⁴ resamples with a
  work-budget cap ≥10³ for huge pools; numpy fast path + stdlib fallback). All 7
  analyzer/table scripts migrated; flat-vector calls still work but are labeled
  `pooled-legacy`. Effect on committed data: E1 CIs widen ~3–4× (point estimates
  unchanged, zero-copy claim survives cleanly); E2 mps exposes contaminated runs
  (§3.2). `table1.tex` / `table2_mps.tex` regenerated.
- [x] **Run-level values printed beside every CI** (`run_medians` in summarize; table
  scripts emit them as LaTeX comments; analyzers as columns).
- [x] **Paired across-run difference implemented** (`paired_overhead_ci`) and made the
  E2 statistic of record; raw per-pair differences printed.
- [x] **p95 gate raised to ≥500 pooled** in both libs; p99 not estimated.
- [ ] **Add CIs to `bandwidth_analysis.py` and `compare_factoid_engines.py`** — deferred
  until the E3′/E4 final experiment shapes land (both scripts will be reworked then);
  no E3/E4 number is reportable from them as-is. Also fix the crude p95 index.
- [ ] **Raise R to 10 for cheap cells** (E1, E2, E3 — minutes per run): runs are the
  replication unit, so more runs beat more queries for interval validity. Keep R=5 only
  where a run is expensive (E4 full arms, E6 cells).

### 2.4 Analysis scripts — **P1 [dev]** (extend what exists)
- [ ] E1: `analyze_noop_results.py`, `analyze_payload_results.py --fig`,
  `generate_latex_results.py` — re-run on the verified M2 Pro data.
- [ ] E2: `analyze_operational_overhead.py`, `true_overhead_analysis.py`,
  `breakdown_overhead.py`, `generate_latex_results.py` — per device.
- [ ] E3/E4: `bandwidth_analysis.py` (timing 2×2) + the new quality scorers + CIs (§2.3).
- [ ] E5: script to confirm the arrival process matches each MLPerf scenario from the
  trace (requires the never-blocking `queue_depth` rule, §2.6).
- [ ] E6: degradation-curve + **per-process attribution** analyzer (GB10 `nvidia-smi pmon`
  per-process SM activity, cross-checked against the aggregate device counter; residency-
  only on M2 Pro).
- [ ] E7: resident-memory-vs-size and quality-vs-size, with the per-DUT ceiling marked.

### 2.5 Framework fixes that gate final collection — **P0 [dev]** — **DONE (working tree, uncommitted)**
Plan: `~/.claude/plans/let-s-do-code-level-*.md`. 9 files changed; all compile; fixes
run against real code paths (not mocks).
- [x] **Polling-policy busy-wait:** ~~`FirstSubmittedPolicy` / `MergePolicy` poll with
  `sleep(0.1)` **and `print` on every poll**~~ — **Done:** replaced the sleep-poll with a
  shared `threading.Condition` (each stage's input queues notify it on `put` via
  `PeekableQueue`; the fan-in policies block on it), prints deleted. Confirmed **all 40**
  decomposed Self-RAG `polling_policy` lines are `FirstSubmittedPolicy` (so the bias hit
  the decomposed arm). Verified **end-to-end**: 57 ms wake vs old ~100 ms floor;
  factoid_decomposed_mlx serial completed clean.
  Files: `peekable_queue.py`, `polling/{polling_policy,first_submitted_policy,merge_policy}.py`, `stage.py`.
- [x] **`classification.py`** — **Done:** `zero_grad()` guarded to `query.split=="train"`
  **and** non-None optimizer (kills the ResNet-50-inference NPE + the wasted val call).
  Unit-verified (val + `optimizer=None` runs); true ResNet e2e needs the ~150 GB ImageNet
  set, so left to E5 setup.
- [x] **`IndexRouter.run`** — **Done:** `dump_model_json()` → `model_dump_json()`.
  Unit-verified (dead path in every collection config).
- [x] **`queue_depth` / arrival-process audit — code-level detector Done:**
  `PoissonLoadScheduler.generate()` now times each `put` and warns at the end when submits
  blocked on a full entry queue (blocked count + total/max lateness) → coordinated omission
  is no longer silent. Unit-verified (silent on unbounded, warns on saturated). The
  `queue_depth`-sizing *rule* itself is config-level (§2.6), still pending.
- [~] **Hygiene:** junk imports in `classification.py` **removed**; `loadgen.py` debug
  `print(loadgen)` **removed**. **Deferred (P2):** the unbounded router retry dicts in
  `binary_router.py` / `monolith_router.py` — re-enabling the commented `del`s risks a
  `KeyError` on a re-accessed key, and the leak is a few bytes/query; not worth the risk
  pre-collection.

### 2.6 Hyperparameter protocol — **P0 [dev/M2]** (NEW — agreed 2026-07-12)

Principle: **every config knob derives from a stated, pre-registered rule, not a
value**; a pilot run per (workload, device) supplies the inputs; a post-hoc check
confirms the rule held. The paper's setup section gets a **per-experiment knob table**
(knob → value → rule → verification), which also serves as Ties's collection spec.

- [ ] **Pilot protocol:** one pilot per (workload, device) to measure serial service
  times and warm-up horizons; pilots excluded from reported data.
- [ ] **Query counts — decouple quality from timing:**
  - Quality (EM/F1, answered rate): proportions need N. At N=10 the Wilson interval on
    8/10 is ~[49%, 94%] — meaningless; at N=30, ±17 points. To separate the headline
    multi-hop gap, run **~100–150 questions per arm in dedicated *serial* quality runs**
    (quality doesn't need the contended regime; wall time is the only cost).
  - Timing: 30–50 queries × R per cell is fine with run-level stats; hold the p95 gate
    (≥500 pooled) or explicitly drop p95 for that experiment (decide now for E3).
- [ ] **Load rate rule:** λ derived from the pilot-measured serial service time of the
  *slowest arm in the comparison*, identical for every arm on a device. Three legal
  operating points, chosen per experiment and stated: below-saturation (~0.5–0.8×
  capacity) for stationary latency; **fixed N-in-flight closed-loop** for contention
  (recommended for E3 — current rate=2.0 vs ~3 s/query is ~6× overload, a
  non-stationary regime where latency measures queue growth, not service); saturating
  for throughput-only (never report per-query latency there).
- [ ] **`queue_depth` rule:** sized so the entry queue never blocks in any reported cell
  (Offline: depth ≥ total samples); verified from the arrival trace — this is what makes
  the E5 "arrival process matches the scenario" check honest.
- [ ] **Warm-up rule:** determined empirically once per experiment (rolling-median
  flatness on a pilot), fixed with margin, stated. Expected: E1 k≈5–10 (current 1-of-101
  is too few); E2 k=200 — **fix the code/prose mismatch (code drops 100, prose says
  200)**; E3 additionally drops + separately reports the ANE first-call compilation query.
- [ ] **Workload-defining params** (max_retries, top-k, max answer tokens, batch size):
  from external precedent (Self-RAG defaults, standard RAG settings), held constant
  across every arm of a comparison, provenance stated. **Sensitivity check on the
  load-bearing one: top_k=10 column on the multi-hop core arms** (the GB10 write-up
  itself admits top_k=5 caps recall — cheap insurance for the headline result).
- [ ] **E6 foreground rate:** pilot-derived so the B=0 baseline sits at ~30–50% of
  capacity (headroom for degradation to show; not a strawman), then **held fixed across
  all B**; generator enforces it.
- [ ] **Lock the derived values into committed configs** before the GB10 pass; knob
  tables drafted for the third-edition setup text (§5.1).

---

## 3. Data collection (per experiment)

> **GB10 status (updated):** treat all pre-existing GB10 Self-RAG data as **suspect
> pending §1.5**; the plan of record is a **single full GB10 collection pass run by
> Ties** after §0 items 2–7 land. E2's cuda half is cheap — re-run it in the same pass
> rather than litigating old-data provenance.
>
> General protocol for every cell: fixed distinct seeds, greedy decoding
> (do_sample=false, temp 0) verified by a repeat-run identical-output check, weights
> & datasets pre-fetched, thermally-throttled runs excluded per a **pre-registered
> threshold** (define it — §6), knobs from the §2.6 rules, env + git commit + config
> hash captured per run (§1.5).

### 3.1 E1 — Framework overhead (NoOp) — **P0 [M2]** — `[?]` possibly already done
- [x] Depth × payload × tracing matrix collected (185 CSVs).
- [ ] `[?]` **Reconcile provenance:** this file previously claimed the matrix came from a
  busy shared Linux box, but `run_matrix_env.txt` records an **Apple M2 Pro** (41.7 ns
  clock) and the working tree holds fresh untracked NoOp CSVs + a regenerated
  `payload_zero_copy.pdf`, with `table1.tex` numbers (83.5→60.4 µs/stage) differing from
  the stale Linux write-up (206.8 µs). If the idle-M2 re-collection already happened:
  mark done, delete/regenerate the stale `framework_overhead.md` (§4.1), and record the
  env capture. If any depth-50-style contention artifact persists, re-collect just those
  cells.
- [ ] Capture the **synchronous** tracing cost from an orchestrated run (the committed
  tracing-on numbers are an async-local-store proxy — self-admitted in both overhead docs).
- [ ] Set the warm-up drop by the §2.6 rule (current 1 of 101 is too few); R→10 (§2.3).

### 3.2 E2 — Modularity overhead — **P0**
- [x] **[M2] RE-COLLECTED (2026-07-13, idle session, R=10, commit 33893eb).**
  All 10 tracing-off runs consistent (89.2–90.7 ms — contamination gone).
  **New headline: paired overhead −113 µs [−460, +172], −0.13% [−0.51, +0.19] —
  statistically indistinguishable from zero at ±0.5% resolution.** (The old
  "+0.55%" was a pooling artifact of the contaminated matrix.) Tracing-on:
  +2.3%/step, consistent across all 10 runs. `table2_mps.tex` regenerated.
- [ ] `[?]` Small quirk to check: fresh baseline arms logged ~1450 steps/run vs
  Choreo's 900 post-warmup (baseline_finetune may ignore --max-batches and run
  the full epoch) — medians are steady-state so the verdict stands, but fix for
  the "identical N" claim before the GB10 cuda half.
- [ ] **[GB10, Ties] Re-run the cuda half** in the main collection pass (prior numbers:
  +49.1 µs / +0.13% core; +1.75 ms / +4.5% tracing-on async) — cheap, and sidesteps
  old-data provenance questions.
- [ ] Fix the warm-up code/prose mismatch (§2.6) before the re-run.
- [ ] Re-run analysis with the run-level paired statistic (§2.3).
- [ ] Regenerate/replace the stale `modularity_overhead.md` (reports cuda while the
  committed table is mps) — §4.1.

### 3.3 E3 — VQA bandwidth contention — **P0 [M2]** — ⚠️ REDESIGN PROPOSED
> **See `CONTENTION_EXPERIMENTS_REDESIGN.md` (E3′).** The 2×2 as designed cannot
> attribute the collapse (queueing-dilution confound) and is mis-proportioned
> (co-runner demands ~2–5 GB/s against an LLM already streaming ~110–170 GB/s).
> Proposal: replace with a cross-device bandwidth dose–response experiment
> (decode tok/s vs. co-runner traffic, prefill as negative control); VQA cut or kept
> as a slim illustration; **VQA-accuracy scorer drops from §2.1 if accepted.**
> **UPDATE (2026-07-13, author):** E3′ and E6′ merge into ONE staged experiment —
> system view (E6′) first, then config-only zoom steps to the mechanism (E3′ cells)
> with the YAML diff shown at each transition (see redesign doc §0.3). Author has
> agreed to E6′ and the combined form; supervisor sign-off pending on the package.
> Items below stand only if the redesign is REJECTED:
- [ ] **⚠ NEW (2026-07-13 pilots): the ANE/CoreML apparatus HANGS.** Both ANE pilot
  cells (`e3_vqa_b`, `e3p_c2_rmax` — CLIPVisionEncoderCoreML) stalled to their 1-h
  timeouts instead of erroring. E3 mapping-B (and E3′ co-runner C2) is unusable until
  diagnosed — suspects: CoreML model compile blocking forever on this macOS, or a
  deadlock in the stage's prepare/dispatch. Affects BOTH E3 variants; investigate
  before any E3 collection.
- [x] Verified: **no E3 data exists** (configs `multimodal_vqa_mapping_{a,b}.yml` ready;
  `bandwidth_analysis.py` ready; nothing run).
- [ ] **Decide the operating point first (§2.6):** recommended — build the N-in-flight
  closed-loop scheduler and run the contended cells at fixed concurrency (stationary),
  instead of open-loop rate=2.0 (~6× overload, non-stationary).
- [ ] Collect the **2×2** (mapping A collocated-GPU / mapping B CoreML-ANE) ×
  (contended / serial) × R (R=10 if cells stay cheap, §2.3).
- [ ] Record the **measured stage-concurrency mean** per cell (confirm overlap actually
  occurs in the contended cells).
- [ ] Capture the **required DRAM-bandwidth / power read** (`powermetrics`/`macmon`) — the
  "bandwidth-bound" claim rests on it; without it the claim downgrades to "consistent with
  contention."
- [ ] Discard the ANE first-call compilation query as warm-up; report it separately.
- [ ] Raise `max_queries` toward the p95 gate (≥500 pooled) **or drop p95 for E3 —
  decide now** (§2.6).
- [ ] Run VQA-accuracy + parity scorers on the outputs.

### 3.4 E4 — Self-RAG topology — **P0 [M2] + [GB10]** — re-collection assumed (§1.5)
Arms: Monolith-9B, Decomposed-3×4B, Monolith-4B (size control), Decomposed-Shared
(logical), + engine overlay (HF / vLLM / Ollama). Tasks: factoid + multi-hop. Both DUTs.
- [ ] **Gate: §1.5 provenance.** Working assumption: the full matrix re-runs post-fix
  with §2.6 knobs on both DUTs.
- [ ] **Split every arm into (a) timing cells** (30–50 queries × R=5, pipelined + serial,
  matched load per §2.6) **and (b) serial quality runs** (~100–150 questions per arm) —
  the quality claims cannot be carried by the timing-cell Ns.
- [ ] **[M2]** Complete the factoid timing 2×2 for all core arms (9B, decomposed, 4B,
  shared) at one matched load; where the queue does not back up at the matched rate,
  report serial-only and state pooled N (pipelined story carried by GB10).
- [ ] **[M2]** Collect **multi-hop** for all core arms — incl. **Monolith-4B on multi-hop**
  (the size control that decides whether the multi-hop quality gap is topology or size;
  **must be core**, configs **created** §2.2).
- [ ] **[GB10, Ties] Full matrix** (BF16, §2.6-derived rate) — factoid + multi-hop, all
  core arms + the 4 new multi-hop controls, pipelined + serial + quality runs.
- [ ] **[GB10, Ties] Engine overlay** (HF vs **vLLM** vs Ollama) on the single-instance
  arms — vLLM is CUDA-only; **[M2]** Ollama cross-platform.
- [ ] **Top-k sensitivity:** one top_k=10 column on the multi-hop core arms (§2.6).
- [ ] **[M2/GB10] E7-merge:** run the size-sweep rungs — configs **created**
  (`factoid_monolith_{0.8b,2b,27b}_{mlx,cuda}.yml`, §2.2/§3.7); 27B is the M2 ceiling.
- [ ] Score all runs with **EM/F1** (+ answered rate, parity, LLM-calls-per-query,
  resident model memory); every speed claim conditioned on EM/F1.
- [ ] Commit/release the raw CSVs (with perf column) + JSONL behind every reported
  number (§7) — the current repo carries only a 10-query smoke pair.

### 3.5 E5 — MLPerf reduction + "isolated inference misleads" — **P0 [M2] + [GB10]**
> **E5 has TWO jobs (framing locked 2026-07-13):** (1) *positioning* — MLPerf's scenarios
> are recoverable as Choreo scheduler configs (the superset); (2) *substantive critique* —
> MLPerf's **isolated single-model** number can **mislead** as a proxy for real
> performance, because it excludes the surrounding pipeline (retrieval, pre/post-process,
> data movement, cross-stage handoff). The evidence is a **contrast**, not a scenario
> matrix: same ResNet-50, measured the MLPerf way (Offline, model in isolation → throughput
> X) vs the Choreo way (full end-to-end pipeline with per-stage occupancy → effective
> Y, inference is only a fraction). E2's breakdown already supports this (dataloader ~34 ms
> vs model ~39 ms). **Keep E5's core the end-to-end / whole-pipeline critique so it stays
> distinct from E6 (inter-pipeline collocation) — E6 is the amplifier, not a duplicate.**
> Scenario coverage: SingleStream + Server exist; **Offline built (§2.2)**; **MultiStream
> dropped** (niche, no workload fit) — table marks it "expressible" without an arm.
- [ ] Instantiate the ResNet-50 scenario configs actually used: SingleStream
  (`OfflineLoadScheduler`), Server (`PoissonLoadScheduler`), Offline
  (`SaturatingOfflineScheduler`). The committed resnet config is the Server-like Poisson
  cell; add the SingleStream + Offline `loadgen` variants.
- [ ] **Isolated-vs-end-to-end contrast run:** ResNet-50 Offline in isolation (throughput
  X) vs the same model in a full Choreo pipeline with per-stage occupancy (effective Y).
- [ ] Enforce the `queue_depth` never-blocks rule (§2.6) so the arrival-trace
  verification is honest (Offline: depth ≥ total samples).
- [ ] Collect each scenario on both DUTs (R=5) and confirm the arrival process matches the
  scenario from the trace.
- [ ] `[?]` **Optional but strong:** run the official MLPerf LoadGen (Offline) on the *same
  DUT* for a within-X% parity number, or explicitly scope it out.
- [ ] Repo hygiene that blocks this experiment: `mlperf/bert_inference.yml`,
  `bert_training.yml`, `retinanet_training.yml` are **0-byte stubs**;
  `retinanet_inference.yml` hardcodes `/home/roba/...` (§7.1). Fix or delete.

### 3.6 E6 — Staged contention experiment — **P0, APPROVED (2026-07-13)**
> **APPROVED by the author:** the staged system→mechanism experiment of
> `CONTENTION_EXPERIMENTS_REDESIGN.md` §0.3 — Stage A (RAG-serve foreground vs
> B index-refresh pipelines, per-process attribution) → B (intensity isolation) →
> C (single-resource co-runners) → D (bare decode, prefill/decode split), each
> transition a single-element config diff shown in the paper. **VQA (old E3) is
> CUT** — the CLIP encode configs survive only as Stage-C co-runner apparatus.
> Knob variants locked: `e3=dose_response`, `e6=rag_indexing` (derive defaults).
Decision upgraded (§1): the paper's title and Intro rest on this; minimal E6 ships.
- [ ] Build the B-sweep generator (§2.2) — independent bg processes, each own dataset stage.
- [ ] Fix ONE foreground workload (EfficientNetV2-S inference, Poisson rate per the
  §2.6 headroom rule, held fixed across all B), ≥500 pooled queries/cell; B=0 is the
  isolation baseline.
- [ ] **[GB10, Ties]** B∈{0,1,2}: capture per-process SM-activity (`nvidia-smi pmon`);
  validate the faithfulness check (per-process shares reconcile to the aggregate counter).
- [ ] (Optional) **[M2]** same curve, attribution **residency-only** (no per-process counter).
- [ ] Report the ceiling B reached; do not extrapolate.
- [ ] Framing rule for the prose: Choreo **measures and attributes** interference; it
  never **manages** it (no cross-pipeline resource control exists) — §5.3.

### 3.7 E7 — Capacity/size sweep — **MERGED INTO E4 (Self-RAG)** — **P1**
No longer a standalone experiment; it is the Self-RAG monolith arm read along the size axis.
- [~] 4B and 9B points come from E4's monolith size control — **re-collected with the
  matrix if §1.5 resolves pre-fix** (quality axis is affected by the retry bug; the
  resident-memory axis is not).
- [ ] **[M2]** Add rungs 0.8B, 2B (fit) and optionally **27B (OOM ceiling)** — one-line
  `model` swaps (shared chat template/architecture → genuinely config-only).
- [ ] **[GB10, Ties]** Run the same added rungs.
- [ ] Report resident-memory-vs-size and **quality**-vs-size inside E4 (latency-vs-size is
  noisy on Self-RAG's retry loop — corroborate quality against the family's published
  capability scores). Keep the sparse 35B-A3B MoE off the dense axis.
- [ ] Produce the literal **config diff** (smallest↔largest rung = only the `model` field)
  as the "no-code" artifact within the E4 narrative.

---

## 4. Analysis, tables & figures — **P0/P1 [dev]**

- [ ] Regenerate **Table: NoOp overhead** (`table1.tex`) from the verified M2 Pro data
  with hierarchical CIs (§2.3) — expect intervals to widen; that is correct.
- [ ] Regenerate **Table: modularity overhead** (`table2_mps.tex`) + the GB10 counterpart,
  paired statistic, hierarchical CIs.
- [ ] **Fig: depth-flatness** (transition cost vs depth) — E1.
- [~] **Fig: zero-copy** ref-vs-copy (`payload_zero_copy.pdf/png`) — regenerated;
  confirm against verified data (§3.1).
- [ ] **Fig/Table: VQA 2×2** — within-mapping serial→contended deltas + occupancy/concurrency
  + DRAM-bandwidth trace — E3.
- [ ] **Table: Self-RAG trade-off** — latency/throughput + EM/F1 per task per DUT, with the
  size and logical controls — E4.
- [ ] **Table: MLPerf scenario→config reduction** (drafted as tab:mlperf-map) +
  arrival-trace confirmation + optional parity line — E5.
- [ ] **Fig: collocation degradation curve** (foreground p95 vs B) + per-process attribution
  — E6 (the headline collocation figure).
- [ ] **Fig: capacity ceiling** (resident memory / quality vs model size, per DUT) — E7.
- [x] **Fig: topology diagram** (`topology_diagram.{png,svg}`) — reuse for E4.
- [ ] **Per-experiment knob tables** (§2.6) for the setup section.
- [ ] Ensure every figure is legible in grayscale and has self-contained captions.

### 4.1 Stale result docs — regenerate or delete (NEW, from audit)
- [ ] `framework_overhead.md` still prints old Linux dev-box numbers (depth-1 ≈207 µs)
  contradicting `table1.tex` (83.5 µs).
- [ ] `modularity_overhead.md` reports cuda while the committed table is mps.
- [ ] `monolith_vs_decomposed_{dgxspark,m2pro}.md` — pending §1.5, mark superseded or
  attach provenance.
- [ ] Rule: result `.md`s are generated (or timestamped + commit-stamped), never
  hand-maintained alongside diverging `.tex`.

---

## 5. Writing (paper sections)

### 5.1 Experimental Setup / Methodology — **P1**
- [x] `experimental_setup.tex` drafted (7 experiments, 3 review rounds → Accept).
- [x] `methodology.tex` exists (rev 2, post-review).
- [ ] **Merge/replace decision executed** (§1) and one canonical section produced,
  incl. the §2.6 knob tables and the §2.3 statistics as actually implemented.
- [ ] Resolve all `[[FILL]]`s in the chosen text (§6).

### 5.2 Results — **P0 (UNWRITTEN) — findings-first ordering**
- [ ] Write §Results from scratch, **findings first**: the three insights lead —
  E3 (heterogeneity gain collapses under contention), E4 (config-only topology
  trade-off, conditioned on EM/F1), E6 (collocation degradation + attribution) —
  then E1/E2 as a short "trust the instrument" pass, then E5 as positioning.
- [ ] Each subsection states the measured value + hierarchical CI + raw run values
  against the claim its Methodology counterpart set up.
- [ ] Lean into the GB10 timeliness angle: a careful two-point study across the
  unified-memory design space (16 GB M2 Pro ↔ 128 GB GB10) on barely-characterized
  hardware.

### 5.3 Other sections — **P1**
- [ ] **Abstract** — write/finalize.
- [ ] **§1 Introduction** — align claims with E5/E6/E7 as collected; frame the thesis as
  *what the instrument lets you learn* (the three insights), not the instrument itself.
- [ ] **Honest-framing sweep (from code audit):**
  - "Zero-copy" = CPython reference passing between threads, validated against a
    `deepcopy` strawman, intra-pipeline only — prose must never claim more than the
    experiment shows (methodology's "lower-bounds serialization cost" hedge is correct;
    keep it everywhere).
  - "Collocation-aware" = separate OS processes co-scheduled with per-process listeners —
    say **"measures and attributes," never "manages."**
  - Thread-per-stage GIL implications stated (already in metrics prose; keep).
- [ ] **§2 Background / Related Work** — MLPerf (Inference scenarios, Tiny), AIBench,
  TPCx-AI, DAWNBench, + the two `rob` TODOs (AISBench; HPI VLDB preprint); position
  \sysname{} as superset-not-competitor (frame the mid-range gap).
- [ ] **§3 The \sysname{} Framework** — ensure `\label{sec:framework}` exists; describe the
  declarative graph, process-per-pipeline / threads-per-stage split (explicitly),
  queue reference-passing, RadT/MLflow tracing.
- [ ] **§7 Conclusion + Future Work** — controlled inter-pipeline study at scale, discrete-
  memory datacenter GPUs, multi-node.
- [ ] Confirm every `\ref`/`\label` resolves (esp. §4/§5 currently label-less per methodology
  header note).

---

## 6. Cross-cutting consistency & reproducibility — **P1**

### 6.1 Resolve every `[[FILL]]` (18 in experimental_setup.tex; more in methodology.tex)
- [ ] **Device SKUs / bandwidth:** M2 Pro core split, GPU cores, ≈200 GB/s; GB10 273 GB/s,
  SM count, CPU cores — the whole "8× capacity / 1.4× bandwidth" framing rests on these.
- [ ] **Software versions:** macOS + mlx/mlx-lm/CoreML Tools/faiss-cpu; torch cu130 stack.
- [ ] **Model pins:** HF commit hashes for Qwen3.5-9B/4B (+ ladder) and CLIP-ViT-L/14;
  the OptiQ-4bit MLX quantization recipe / bit-layout.
- [ ] **Dataset pins:** rag-mini-wikipedia, HotpotQA/FlashRAG split, OK-VQA, COCO-Karpathy —
  version + snapshot date; FAISS/ChromaDB index type/metric/build params.
- [ ] **Rates / run counts / seeds** stated per experiment — now derived via §2.6 rules.
- [ ] **Resource-listener sampling period** (macmon / nvidia-smi / dcgmi).
- [ ] **Thermal-throttle policy** — pre-registered threshold + discard-vs-report rule.
- [ ] **Released-artifact URL.**

### 6.2 Consistency
- [ ] **Framework name** unified everywhere (§1).
- [ ] **Reconcile methodology.tex with experimental_setup.tex** — methodology still says the
  collocation experiment is future work; this plan delivers it (E6, now non-negotiable).
  Retire that paragraph.
- [ ] **Reviewer D2 (size confound)** — confirm the Monolith-4B control is run and reported
  (E4 + E7, both tasks); remove the `[[FILL: confirm control arm run]]`.
- [ ] **Setup-text ↔ analysis-code consistency check** — the setup text must describe the
  statistics the scripts actually compute (post-§2.3), and the knob tables must match the
  committed configs.
- [ ] **Citations** — add the ~13 required `.bib` entries listed at the bottom of
  methodology.tex (SQuAD, HotpotQA, VQA, EfficientNetV2, Imagenette, CLIP, FAISS, COCO,
  Karpathy, OK-VQA, Qwen3.5, Self-RAG, Chroma) + MLPerf/AIBench/TPCx-AI/DAWNBench/MLX/CoreML/
  Perfetto/MLflow/GB10 datasheet.
- [ ] Add `\usepackage{wasysym}` (or pifont) for the `\CIRCLE/\LEFTcircle/\Circle` marks.
- [ ] Confirm the section compiles standalone (tables, macros, siunitx).

---

## 7. Artifact & reproducibility — **P1**

- [ ] Decide what to **commit vs gitignore** (results CSVs / JSONL, generated tables/figures).
  Rule from §1.5/§3.4: **every reported number's raw trace + JSONL is committed or in the
  data release** — the current repo backs the headline Self-RAG numbers with nothing but a
  10-query smoke pair.
- [ ] Update the top-level README (env filename `macos.yaml` not `.yml` — same for nvidia;
  env name `benchmark_macos`; the pinned-env rationale) — see REPLICATION_NOTES.md.
- [ ] Pin the conda envs (macOS + nvidia + engines) with the exact resolved versions used
  for the reported runs; document the radt `async_tracing` branch (commit 3ba61cb).
- [ ] One-command reproduce script per experiment (collect → analyze → table/figure).
- [ ] Prepare an **artifact-evaluation** package: released traces + scripts that regenerate
  every reported number, with a top-level "how to reproduce Table/Fig N" map.
- [ ] Sanity-check a clean clone reproduces at least the M2 Pro spine end-to-end.

### 7.1 Artifact landmines from the code audit (fix cheaply, early) — **P1**
- [ ] Hardcoded absolute path `/home/roba/...` in `pipeline_configs/mlperf/retinanet_inference.yml`.
- [ ] **0-byte config stubs:** `mlperf/bert_inference.yml`, `bert_training.yml`,
  `retinanet_training.yml` — populate or delete.
- [ ] `pipeline_configs/self_rag.yml` references a missing `tmp/accounting.sqlite` —
  ship the DB, a builder script, or drop the config.
- [ ] ANEMLL submodule failure mode: `stages/anemll` imports break without
  `--recurse-submodules` — document + guard with a clear error.
- [ ] **33 MB `mlflow.db` + ~130 `mlruns/` dirs committed** — gitignore + purge from the
  artifact (they also risk shadowing fresh runs via reused experiment id 0).
- [ ] `main.py` writes `evaluation/results/` relative to CWD — must be run from repo
  root; document or make robust.
- [ ] Untracked-vs-committed drift: PAPER docs, REPLICATION_NOTES, result CSVs currently
  untracked; the committed tree differs from what produced the current numbers.

---

## 8. Submission logistics — **P1/P2**

- [ ] Confirm venue + deadline against GB10 re-collection speed (§1 ladder: ASPLOS →
  MLSys → EuroSys/ATC → ISPASS/IISWC); fit template + page limit.
- [ ] Anonymize for review (strip author/URL/machine identifiers; anonymized artifact).
- [ ] Internal read-through against a reviewer rubric (claims↔evidence, confounds, stats).
- [ ] Passes: spelling/grammar, figure legibility, `\ref` integrity, `.bib` completeness.
- [ ] Author list, affiliations, acknowledgements, funding.
- [ ] Submit + upload artifact.

---

## Appendix A — Blocker taxonomy (post-audit, 2026-07-12)

**Gate zero (decides the whole schedule):** the `query_id` provenance of the headline
Self-RAG data (§1.5) — working assumption is full GB10+M2 E4 re-collection.

**Hard (a claim is unreportable without it):** EM/F1 + VQA-accuracy scorers (§2.1);
hierarchical bootstrap + E3/E4 CIs (§2.3); post-fix E4 data on both DUTs (§3.4);
minimal E6 (§3.6 — now non-negotiable); E1 provenance verification (§3.1 — possibly
already satisfied).

**Build work (before the GB10 pass):** ✅ framework fixes (§2.5); ✅ saturating-Offline
scheduler; ✅ 4 multi-hop control configs; ✅ E7 size rungs (all §2.2, done 2026-07-13);
MultiStream **dropped**. *Remaining:* N-in-flight scheduler + E6 B-sweep generator (both
contingent on the E3′/E6′ redesign); §2.6 pilots + locked knobs.

**Smaller:** E3 DRAM-bandwidth read (§3.3); synchronous tracing cost (§3.1); top-k
sensitivity column (§3.4); stale result-doc regeneration (§4.1); artifact landmines (§7.1).

## Appendix B — Status snapshot (post-audit, 2026-07-12)

- **Decisions locked:** name = **Choreo**; setup text = third edition; scope = E1–E5
  core, **E6 minimal NON-NEGOTIABLE**, E7→merged into E4; **GB10 = full re-collection
  by Ties** (old "data exists, just analyze" superseded pending §1.5); venue = ASPLOS
  stretch with MLSys/EuroSys/ATC/ISPASS-IISWC ladder.
- **E1 NoOp:** 185-CSV matrix collected; provenance contradiction (env file says M2 Pro;
  this doc previously said busy Linux box) → verify, likely done; sync-tracing cost +
  warm-up rule outstanding.
- **E2 modularity:** mps done (+0.55%); cuda half to re-run in the GB10 pass (+0.13%
  prior); warm-up code/prose mismatch to fix.
- **E3 VQA:** confirmed **never run**; apparatus ready; operating-point decision
  (N-in-flight recommended) + powermetrics read pending. Cheapest uncollected insight.
- **E4 Self-RAG:** headline write-ups exist for both DUTs but raw data is off-repo,
  committed CSVs lack perf column, and **all of it is suspect pending the query_id
  provenance check** — assume full re-collection with §2.6 knobs, split timing/quality
  runs, + 4 multi-hop controls + top-k sensitivity + EM/F1.
- **E5 MLPerf:** Offline scheduler **built** + SingleStream/Server exist; MultiStream
  **dropped** (niche); SingleStream/Offline resnet `loadgen` variants to add; framing
  locked = superset + "isolated inference misleads" contrast (§3.5); 3 configs are 0-byte
  stubs; one config has
  a hardcoded home path; queue_depth rule needed for the trace check.
- **E6 collocation:** not built; **ships in minimal form** — generator + GB10 B∈{0,1,2}
  with per-process attribution.
- **E7 capacity:** merged into E4; rungs to add; memory axis unaffected by the retry
  bug, quality axis re-collected with E4.
- **Framework:** collection-gating fixes **DONE** (working tree, uncommitted, §2.5) —
  polling busy-wait → blocking Condition wait (verified e2e), inference-path `zero_grad`
  guarded, `IndexRouter` typo, coordinated-omission telemetry, hygiene. Router-dict
  cleanup deferred. Not yet committed.
- **Statistics:** hierarchical bootstrap not yet implemented (setup text currently
  promises what the code doesn't do); E3/E4 analyzers have no CIs.
- **Writing:** methodology.tex (rev2) + experimental_setup.tex (v4) exist; **third
  edition + Results unwritten** (Results to be written findings-first); Intro honest-
  framing sweep (zero-copy / measures-not-manages) pending.

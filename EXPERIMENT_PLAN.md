# Choreo — Experiment Execution Plan (path to full collection)

**PLAN FOR REVIEW. No implementation or collection until approved.** Companion to `EXPERIMENTS.md`
(definitions). Per experiment: **thesis · done · gap · code TODOs · data points · commands.**

**Locked decisions (2026-08-17):**
- **Five** core experiments (E1–E5); **capacity sweep = stretch**.
- **Tracing is infrastructure, not a contribution.** Present **no** old tracing numbers, **no** old
  radt. **Recollect everything with the new bulk+proc tracing** (`CHOREO_PROC_TRACE=1`). res17 exp
  **138** (throwaway 142 was prototype-only).
- **Dead-simple execution.** Collection = a bash `for`-loop over `python main.py <cfg>`. Configs are
  **static, fully-explicit YAML, one per (variant × device)** (device/`serialize_queries`/loadgen/
  listeners all in the config — **no CLI flags**). radt does all orchestration (no `-p` → it schedules
  single- or multi-pipeline configs alike). Python only for **config generation (once)** and
  **analysis**. Retired: `run_collection.py`, the `validate_pass.py` gate, and `generate_latex_results.py`.

The canonical loop (one `collect.sh` per experiment):
```bash
#!/usr/bin/env bash
set -euo pipefail
export CHOREO_PROC_TRACE=1                 # new bulk+proc tracing
# (E5 only also exports RADT_PRESENT + the per-device RADT_LISTENER_* for attribution)
DEVICE=${1:-cuda}; RUNS=${2:-5}; EXP=${3:-138}
OUT="results/$DEVICE"; mkdir -p "$OUT"
for cfg in configs/*_"$DEVICE".yml; do
  name=$(basename "$cfg" .yml)
  for r in $(seq 1 "$RUNS"); do
    python main.py "$cfg" -e "$EXP" --label "${name}_r${r}"   # no -p → radt orchestrates
  done
done
```

---

## REVIEWER PANEL (2026-08-22) — four independent ASPLOS reviews. Verdict: REJECT (2/5 x4)

Four fresh reviewers, no shared context, each given a different lens and the repo. Every
technical claim below was re-verified locally before being recorded. Scores: novelty 2/5,
rigour 2/5, methodology 2/5, contribution/venue 2/5. All four: reject, resubmit.

### VERIFIED errors in OUR analysis (all flattered the result)
1. **E2's headline understates framework cost ~8x.** `stages/stage.py:301-332` brackets
   `run,start -> run,end` around `self.run(query)` only; `_get_input_from_queues()` is called
   at line 343 in `run_wrapper`, **outside** the timed region. So the queue pop + thread wake
   — the 25-39 us/stage hand-off E1 exists to measure — is excluded from E2 **by construction**.
   End-to-end from the same CSVs (b8, 10 runs/arm): cuda baseline **15.13 steps/s** vs choreo
   t0 **14.68** = **-2.9% (~2.0 ms/step)** against the +257 us we report. mps is ~0 (7.707 vs
   7.719). **Report end-to-end throughput alongside per-step — it is in our CSVs already.**
2. **The bf16 test refutes the mechanism it was cited to confirm** (see E4 section below).
3. **Prefill 11-15x cells are M2 GPU oversubscription, not device compute.** The `decomposed`
   arm holds 4 separate 4B models, each with its own lock, so nothing serialises them: median
   **3.00 co-runners** during prefill on those cells vs 1.00 everywhere else, and 7303/3.00 =
   2435 ms ~ the uncontended M2 4B prefill. Uncontended cells only: **6.13/6.16/6.19x (4B),
   8.04/7.97x (9B)**. The honest range is ~6x/~8x, not 5.8-15x.
4. **tok/s was biased N/(N-1)** — 2.00x for 2-token mlx arms vs 1.33x for 4-token cuda ones,
   so it did not cancel cross-device. Fixed.
5. **Up to 55% of stage time was in neither phase** (mutex wait precedes the first_token start
   marker). Fixed: surfaced as an "unaccounted" column.
6. **Decode compared per call, not per token.** Fixed.

### VERIFIED code defects
- `stages/stage.py:231` compares the polling policy against `"stages.queues.polling.
  SingleQueuePolicy"` but the schema default (`utils/schemas/stage.py:14`) is
  `"utils.queues.polling..."` — **the fan-in guard can never fire**; a misconfigured
  multi-input stage silently starves upstreams instead of raising.
- `stages/llm_server/inference.py` has **zero** prefill/decode markers — E4's instrument does
  not exist on the vLLM/Ollama backend, i.e. exactly the backend a serving reviewer demands.
- Every NoOp config sets `disable_logs: false`, so E1's "core dispatch" **includes our own CSV
  trace logging**. There is no `disable_logs: true` arm. One flag, missing.
- **vLLM configs exist and were never run** (`factoid_{monolith_4b,decomposed_shared}_vllm_cuda.yml`,
  0 collected runs). The better backend was available the whole time.

### Design-level objections
- **Novelty:** the prefill/decode asymmetry is textbook (Shazeer'19; Pope MLSys'23; Orca; vLLM;
  FlexGen; Sarathi-Serve; DistServe) and **Splitwise (ISCA'24) already published the
  cross-hardware form and built a system on it**. The bf16 result restates the founding premise
  of GPTQ/AWQ/QLoRA. "Throughput-only benchmarks can't see this" is a strawman: MLPerf's LLM
  server scenario imposes TTFT/TPOT constraints and vLLM's `benchmark_serving.py` reports
  TTFT/TPOT/ITL by default.
- **The zero-copy arm is tautological** — the `ref` stages never touch the payload
  (`stages/stage.py` `run()` just forwards the reference), so O(1) is true by construction.
  `copy.deepcopy` per hop is also not what any in-process framework does.
- **No comparison to ANY existing framework** (Ray Serve / Triton / raw `queue.Queue`). "Our
  overhead is low" is an assertion, not a comparison. ~40 lines would fix it.
- **E2 statistics:** 14/18 cells have CIs containing zero; 6/18 negative; the analyzer prints
  `TOO NOISY to call fixed` four times against a thesis asserting the opposite; the t2-t0
  tracing contrast is computed and never printed (mps b64 = -1648.6 us, CI entirely negative —
  a second apparatus signature).
- **E4 flip:** rests on 1 task at R=1 — data our own warm-up rule says to discard — at 2-3x
  different offered load, with 2 of 8 cells truncated (28/110 and 51/110 completed). Mean-
  while a **cleaner task-flip with 3 replicates per side already exists in the GB10 data** and
  we do not claim it.

### What survives review
E1 core dispatch (46 cells x R=10, R^2=0.99996, slope robust to dropping low depths);
decomposition is prefill-amplifying (79-84% -> 97-98%, within one device+engine); ~6x/~8x
prefill speedup on uncontended cells; **the marginal bandwidth term is device-invariant
(193 vs ~191 GB/s implied) while a ~34 ms/token CUDA software constant explains the decode
gap** — a better result than the one we claimed; E3's parity-before-critique; and the
measurement discipline itself (artifacts caught and root-caused rather than reported).
Reviewer 2 independently verified: sampling matched, token budgets matched, 0 unpaired
markers across all 32 traces, cuda replicates <1%, and **no thermal drift**.

### Recommended reframe (panel consensus)
Lead with what is actually novel and currently buried: **cross-role prefill redundancy that
deployed prefix caches miss by construction**, the **token-count vs joules inversion**, and
**topology as a phase-mix knob** (0.27 -> 0.65 prefill share from a YAML edit). Venue: MLSys
or ISPASS now; ASPLOS only if E5's engine-specific contention is collected on calibrated
counters and made the centerpiece.

---

## P0 — prerequisites (GATE ALL COLLECTION)

**A. Deploy the new tracing.**
1. Vendor/pin **bulk radt** (`itu-rad/radt @ feat/proc-owned-bulk-tracing`) at a fixed tag; install
   into **both** envs (`benchmark_macos`, `benchmark_nvidia`), replacing `@9dda7b8`; update
   `environments/{macos,nvidia}.yaml` + the pin the drivers assert.
2. **Commit** the Choreo proc wiring (`utils/trace_span.py`, `main.py` hooks, 12 span sites) on the
   collection branch (else a dirty tree blocks nothing now, but keeps churning).
3. Fix the **listener/SIGTERM** interaction (add a main-PID guard in `_on_sigterm` so forked listeners
   don't try to join the trace-exporter child).
4. **Smoke bulk+proc on every workload family** (Self-RAG, staged, ResNet, 3D-UNet) at exp 142:
   rc=0, `radt-trace/spans-*.jsonl.gz` + `manifest.json` land with full `event_count`, listeners log,
   no jitter.

**B. Build the minimal harness (replaces `run_collection.py`).**
5. Adopt the **one-YAML-per-(variant×device)** rule: convert any config that relied on runtime
   rewriting (device patch, loadgen override, serialize) into explicit static YAMLs.
6. Per experiment write: a **`gen_configs.py`** (only where a sweep needs many files — writes YAML,
   run once), a **`collect.sh`** (the loop above), and an **`analyze.py`** (results-dir → plots/tables).
7. Confirm `python main.py <cfg>` (no `-p`) is the single entry point for both single-pipeline and
   multi-pipeline (radt-orchestrated) configs; keep `-p 0` only where an experiment wants to skip the
   orchestrator (the overhead cells, optionally).

**Exit gate:** four green workload smokes + the harness scaffolding in place. Nothing collects before.

### P0 progress (2026-08-17)
- **A.1 DONE.** Bulk+proc radt pinned `@0b497f6` in both `environments/{macos,nvidia}.yaml`;
  editable-installed into `benchmark_macos` (mlflow → 3.15.1, matching GB10, which was already on it).
  Modularity env gate now also asserts `radt.trace` exists (0.2.29 alone can't tell bulk from the old
  async fork). Pinned checkout: `~/Documents/work/research/radt-bulk` (branch `pinned-0b497f6`).
- **A.2 DONE.** Proc wiring committed on this branch (`utils/trace_span.py`, `main.py` hooks, 12 span
  sites).
- **A.4 GREEN (overhead family, all three paths).** NoOp depth-16 via bulk+proc: (a) direct `-p 0`
  local file store and (b) orchestrated via `collect.sh` (no `-p`, macmon listener) uploading to res17
  exp 142 — both rc=0, deterministic `event_count=10540` (2×5270 spans), one gzipped
  `radt-trace/spans-000001.jsonl.gz` + `manifest.json` (confirmed *on res17* for the inner run), no
  dropped events, no SIGTERM assertion, CSV timing intact, CSV sorted into `evaluation/results/mlx/`.
  Minor teardown cosmetic: macmon prints "Broken pipe" when its stdout closes on listener teardown —
  non-fatal (run + upload succeed). **Still to smoke:** 3D-UNet / Self-RAG / collocation families
  (need KiTS19 + LLM weights + GB10).
- **A.3 DEFERRED.** Listener/SIGTERM assertion is non-fatal (teardown-only, data intact) and only
  fires on a *killed* run; will fix in radt source only if the res17 orchestrated smoke shows it
  actually recurs on bulk radt (must then sync GB10 + upstream).
- **B.6 DONE (scaffolding).** Canonical `evaluation/collect/collect.sh` — `<config-glob> <device>
  [runs] [exp]`, sets `CHOREO_PROC_TRACE=1`, loops configs×runs over `main.py` (no `-p`), sorts CSVs
  into `evaluation/results/<device>/`. Per-experiment `gen_configs.py`/`analyze.py` still per E1–E5.

### Overnight collection (2026-08-18)
- **E1 NoOp FULLY COLLECTED, both devices, all green.** 23 configs × {tracing-**proc**, tracing-**off**}
  × R=10 = **460 runs/device (920 total)**, exp **138**, forced **bulk** backend (`RADT_TRACE_BACKEND=radt`),
  direct `-p 0`. **0 failures, 0 empty CSVs.** Sanity-gated on a verified res17 `manifest.json`.
  Wall: mlx ~97 min, cuda ~61 min. CSVs: `evaluation/results/mlx/` (local, 225 MB) and
  `…/cuda/` (on GB10 — **still needs pulling for analysis**; spans for both are on res17 exp 138).
  Note: proc & off arms log identical CSV markers (markers are span-independent); the arms differ in
  per-step *timing* + span capture — that's the tracing-overhead comparison E1 makes.
- **P0 family smokes green (exp 142), both devices:** overhead ✅, **3D-UNet ✅** (mlx 4-case 380s,
  bulk `event_count=7224`/3 batches; cuda ✅), **Self-RAG ✅** (0.8B/5-query, bulk finalized). That's
  3 of the 4 P0 families; **collocation (E5) smoke still pending.**
- **Two issues found:** (1) **Self-RAG teardown race** — a span is emitted after `radt.trace`
  shutdown closes the queue (`ValueError: Queue is closed`) on the multi-threaded LLM workload;
  non-fatal (manifest still writes, smoke green) but **harden before full E4 *timing* collection.**
  (2) Operational: background jobs get reaped in this env — collection was driven/finished via
  foreground calls; completed runs persist their CSV + bulk artifact regardless.

### E1 analysis + cuda re-collection (2026-08-18)
- **E1 analysis consolidated** into a single self-contained `analyze_e1.py` (tables + `--latex DEVICE`
  + figures); removed `noop_lib.py`, `analyze_noop_results.py`, `analyze_payload_results.py`,
  `generate_latex_results.py`.
- **cuda E1 RE-COLLECTED clean.** Pass-1 (unpinned) depth sweep was noisy (bimodal `O(d)`); root cause
  = the heterogeneous **Grace** scheduler bouncing threads across cores → cache-cold migration. Pass-2
  (10× X925 cluster `taskset -c 5-9,15-19`) fixed the transition cost but `O(d)` stayed bimodal
  (intra-cluster migration). **Pass-3 = single performance core (`taskset -c 19`, cpu19 @ 4.0 GHz)**
  is pristine: core-dispatch **25.15 µs/stage**, razor-tight CIs, no bimodality. This is the canonical
  cuda E1. Backups on GB10: `cuda_pass1_noisy`, `cuda_pass2_cluster`. **mlx was already clean unpinned**
  (macOS scheduler kept the closed loop stable) — a methodology asymmetry to note in the paper; strict
  symmetry would need mlx single-core-pinned too (awkward on macOS: no `taskset`).
- **Headline (tracing OFF, core dispatch):** mlx **39.4 µs/stage**, cuda(X925) **25.2 µs/stage**;
  zero-copy `ref` flat (mlx ~26 µs / cuda ~14 µs) vs deep-copy at 10 MiB (mlx 1331 µs = 52×, cuda 703 µs
  = 50×). Both: `ref` O(1), `copy` O(payload). **Caveat:** the cuda tracing-ON (proc) arm is slightly
  noisier because the exporter child shares the single pinned core; the OFF-arm dispatch is the clean
  headline. A clean cuda tracing-add would need workload + exporter on *separate* cores.
- **OPEN: res17 `413`** — nginx rejects the bulk span artifact for deep pipelines (≥ depth ~100);
  confirmed `depth_100` landed 0 span artifacts. Does **not** affect any E1 number (CSV-based). Fix TBD:
  raise res17 `client_max_body_size` **or** chunk radt's bulk upload. Real workloads (E2+) have far
  fewer spans/run so likely unaffected.

---

## E1 — NoOp / framework overhead
- **Thesis.** dispatch small + flat; context passing zero-copy.
- **Done.** apparatus + (invalid) 185-CSV matrix.
- **Gap.** re-collect on **both devices** (idle M2 + quiesced GB10) with new tracing; port to the bash
  model.
- **Hyperparameters (pilot-tune).** `max_queries`/run (enough for a stable median, ~100) and the
  warm-up queries dropped; R (5). *Fixed axes* (not tuned): depth set {1..128}, payload sizes
  {0,1 KB,1 MB,10 MB}, ref/copy.
- **Code TODOs.** `gen_configs.py` (depth {1..128} + payload {0..10 MB} × {ref,copy} × tracing
  {off,on} → static YAMLs — likely already exist as configs, just freeze them); `collect.sh`;
  `analyze.py` (depth-flatness line, zero-copy line, headline table).
- **Data points.** all NoOp configs × **R=10** × `{mps, cuda}` (idle M2 + quiesced GB10).
- **Commands.**
  ```
  python evaluation/overheads/framework_overhead/gen_configs.py          # once, if needed
  bash   evaluation/overheads/framework_overhead/collect.sh mps  10      # idle M2
  bash   evaluation/overheads/framework_overhead/collect.sh cuda 10      # quiesced GB10
  python evaluation/overheads/framework_overhead/analyze.py results/{mps,cuda}
  ```

## E2 — Modularity overhead (model × batch sweep)
- **Thesis.** wrapping real work is negligible; relative overhead amortizes with scale, absolute fixed.
- **Done.** `gen_configs.py` → static per-(cell × device) YAMLs; `collect_e2.sh` (three
  configurations, order rotated by repetition, provenance header, timestamped log under
  `collect_logs/`, local MLflow store); `analyze_e2.py` (self-contained: tables, `--latex MACHINE`,
  figures); `baseline_finetune.py` rebuilt so its logging instrument matches `main.py`'s exactly
  (single synchronous `FileHandler`, no listener thread, no stderr sink — the monolith is now an
  actual single-threaded loop).
- **In flight.** Collection on both machines, started 2026-08-27.
- **Metric of record: TIME PER QUERY** — start-to-start between consecutive queries, i.e.
  1/throughput, covering the whole cycle including data loading. Anchor-invariant in steady state,
  which is what makes the monolith and Choreo comparable although they emit different markers. It
  replaces the in-step training-marker comparison, which excluded by construction the between-step
  interval where the framework's cost actually lands, and so measured a near-zero difference
  against ±600 µs of run-to-run noise.
- **Co-headline: the query latency breakdown** from the `choreo-traced` spans — per-stage latency
  (dataloader, training) plus the auxiliary framework overheads (entry, handoff, exit, turnaround).
  Successive instants within one query on one clock: non-negative by construction, no run-level
  term, and they sum to the time per query exactly. Negative intervals are a hard failure.
- **Hyperparameters.** 300 steps/run, warmup 50 — step time is flat from step 0 on both machines,
  so the old 1000/200 was never buying within-run settling. **R = 11 collected, run 1 dropped as
  system warm-up → 10 usable**; the paired statistic bootstraps over runs, so repetitions buy more
  than steps. *Fixed:* batch {1,2,4,8,16,32,64} at EfficientNetV2-S, plus M and L at b8 (ConvNeXt-L
  dropped); fine-tune head; **`num_workers=0`** — a control (not a tuned knob) that removes the
  concurrent-prefetch data path, applied identically on both sides. Both Choreo configurations run
  `disable_logs: true` so no synchronous write+flush sits inside the measured interval on one side
  only.
- **Data points.** `{monolith, choreo, choreo-traced}` × 9 cells × `{m2pro, gb10}` × **R=11**.
- **Commands.**
  ```
  python evaluation/overheads/modularity_overhead/gen_configs.py
  bash   evaluation/overheads/modularity_overhead/collect_e2.sh m2pro 11
  bash   evaluation/overheads/modularity_overhead/collect_e2.sh gb10  11
  python evaluation/overheads/modularity_overhead/analyze_e2.py
  ```
- **Caveats to state in the paper.** `serialize_queries: true` means E2 measures the framework with
  pipelining disabled — the configuration where it can only lose. Choreo runs 6 threads to the
  monolith's 1; that is genuine modularity cost and should be named rather than left implicit.

### E2 — OPEN (deferred, not blocking)
- **Arm-ordering bias.** `collect.sh` runs the arms in a fixed order within each
  repetition (baseline -> t0 -> t2), so any within-repetition warm-up/clock ramp
  systematically penalises whichever arm goes first — the baseline. Suspected cause of
  the **EfficientNetV2-L b8 mps** result: core = **-575.8 us [-737.8, -420.6]**, i.e. a
  *consistent* negative across all 3 runs (-514.5 / -575.8 / -628.7) with a tight CI.
  A wrapper cannot make work faster, so this is apparatus, not a finding. Cheap test:
  alternate the arm order across repetitions (baseline-first on odd runs, t2-first on
  even) and see whether the negative disappears. Until then, do NOT present Eff-L b8 mps.
- **Bistable cells (mps small batch).** effv2s b2/b4 land in one of two step-time regimes
  per process launch (~46 ms vs ~105 ms at b4, same config). `--max-regime-ratio 1.25`
  drops repetitions whose arms straddle regimes, but too few valid pairs remain. Exclude
  the mps small-batch cells from the amortization claim and say why.

## E3 — MLPerf / 3D-UNet
- **Thesis.** (1) **match MLPerf's own reference harness on GB10** — accuracy (Dice) AND performance
  (latency/throughput), *same device* → clean parity, proves Choreo reproduces the MLPerf setup
  (GB10 only). (2) its **offline-preload boundary misleads online** (both devices): preprocessing/
  loading is on the per-request critical path (nothing to prefetch for a fresh request), a real,
  sample-variable, device-dependent share of latency. **Not** a hiding story.
- **Done.** 3D-UNet pipeline + configs `pipeline_configs/unet3d_kits19_{cuda,mlx}.yml`; R=1 CSVs both
  devices; `analyze_preprocessing.py`, `FINDING_A.md`; MPS-validated. (`run_pipelined.py` is the old
  hiding analysis — out of scope.)
- **Gap.** (a) **run the MLPerf reference harness on GB10** for the ground-truth numbers + build the
  Choreo-vs-reference **parity** (accuracy + performance, same device); (b) R=5; (c) new tracing;
  (d) bring the Choreo run onto `main.py + collect.sh` (the 42 cases are iterated by the pipeline's
  case-loader, so `python main.py <unet3d_cfg>` collects them all).
- **Code TODOs.** stand up the **MLPerf reference harness on GB10** (`scratchpad/mlperf-inference`) and
  record its accuracy + latency/throughput; a **Dice scorer** for the Choreo outputs; `collect.sh`;
  `analyze.py` = the **parity table** (Choreo vs reference on GB10 — accuracy + performance) + the
  **preprocessing/loading fraction of per-request latency** (per sample, per device).
- **Hyperparameters (pilot-tune).** R; the **serving scenario** — SingleStream / online, one request
  at a time (the regime where preprocessing is exposed on the critical path; NOT offline batch);
  batch = 1. *Fixed by MLPerf (must match to reproduce):* ROI 128³ + 50% sliding-window overlap,
  preprocessing (resample [1.6,1.2,1.2], HU clip [-79,304], normalization), the 42-case set, the Dice
  accuracy gate.
- **Data points.** *Prong 1 (parity):* MLPerf reference + Choreo, **GB10 only**, R=5. *Prong 2
  (boundary):* Choreo 3D-UNet 42 cases × `{mps, cuda}` × R=5. (ResNet scenario-reduction: **cut**.)
- **Commands.**
  ```
  # prong 1 (GB10): run the MLPerf reference harness for ground-truth, then Choreo on the same box
  ( cd scratchpad/mlperf-inference/... && <mlperf reference run on GB10> )   # accuracy + perf
  bash   evaluation/unet3d/collect.sh cuda        # Choreo, GB10
  bash   evaluation/unet3d/collect.sh mps         # Choreo, M2 (prong 2 cross-device only)
  python evaluation/unet3d/analyze.py results/{mps,cuda}   # parity (GB10) + preprocessing-fraction of latency
  ```

### E3 progress (2026-08-20)
- **Harness built.** Static 42-case configs per device (`evaluation/unet3d/configs/unet3d_42_{cuda,mps}.yml`)
  in the ONLINE regime (`serialize_queries`, `queue_depth 1`, batch 1 = MLPerf SingleStream, the
  regime where load+preprocess cannot be prefetched); `evaluation/unet3d/collect.sh`; and the
  single-file `evaluation/unet3d/analyze_e3.py` (parity table + boundary table + 2 figures).
- **MLPerf reference IS runnable on GB10** — loadgen built in `benchmark_nvidia`, 44 preprocessed
  MLPerf cases + `preprocessed_files.pkl` present, model present. Prior GB10 logs were **Server**
  scenario (from the old scheduling work); E3 uses **SingleStream**.
- **Sequenced GB10 run** (`scratchpad/overnight/e3_gb10.sh`, GPU is exclusive so never overlapped):
  MLPerf perf → MLPerf accuracy (+`accuracy_kits.py` Dice) → Choreo 42-case R=5 → Choreo stage-code
  Dice (`run_full_experiment.py`, same code path as the pipeline stages).
- **CAVEAT to state in the paper:** the reference run uses `min_query_count = 43` (one QSL pass)
  instead of the 1024 an official SingleStream submission requires — at ~8 s/query that would be ~3 h.
  This is a **same-device parity check, NOT a compliant MLPerf submission.**
- **Cost measured:** GB10 ~8 s/case inference; mps ~84 s/case (42 cases ≈ 59 min/run), so **mps is
  capped at 2 runs** — the preprocessing share is a within-request ratio and stable across
  repetitions, and 42 cases already give the sample-to-sample variation prong 2 needs.
- **Prong-2 signal already visible (mps, first cases):** load ≈ 0 ms (the loader only resolves paths;
  the NIfTI read happens inside preprocess), preprocess 3.9–10.4 s, inference 64–139 s →
  preprocessing share **5.8–9.9 %** of per-request latency on the M2.

## E4 — Self-RAG decomposition & prefill/decode split
- **Thesis.** decomposition = prefill/decode rebalance; optimum flips across devices.
- **Done.** 4 arms × factoid/multihop × cuda/mlx configs; **quality validated** (16 cells, greedy
  R=1, judge overturn=0); `stage_latency.py`.
- **Gap.** (a) build the **prefill/decode extractor** + the **cross-device flip** figure (markers are
  in the traces; the analysis isn't built); (b) re-collect **timing** at full-R with new tracing
  (quality does NOT need re-run — greedy); (c) port to bash model.
- **Code TODOs.** prefill/decode extractor (TTFT vs decode per stage per device → prefill:decode
  ratio); flip figure; `collect.sh`; keep EM/F1 + Haiku judge as offline analysis.
- **Hyperparameters (pilot-tune).** the **retry budget / max retries** (the self-correction loop — the
  mechanism, tune per task difficulty); retriever `top_k` (5) + embedder; grader relevance/hallucination
  decision thresholds; decode max tokens; arrival λ for the timing cells; R (timing); warm-up;
  `n_quality` (120). *Fixed:* greedy decoding (deterministic → quality R=1), prompt templates, model
  sizes 4B/9B.
- **Data points.** timing: 4 arms × {pipe, serial} × R=5 × `{mlx, cuda}` (pipe/serial are two
  explicit configs — `serialize_queries` in the YAML). quality: already valid; re-run only on config
  change.
- **Commands.**
  ```
  bash   evaluation/self_rag/collect.sh mlx  5       # for cfg in configs/*_mlx.yml; main.py ...
  bash   evaluation/self_rag/collect.sh cuda 5
  python evaluation/self_rag/analyze.py results/{mlx,cuda}    # prefill/decode flip + decomp cost
  # quality (only if configs changed):  Workflow  llm-judge-selfrag
  ```

### E3 — compliant MLPerf run: DISCARDED, needs a rerun (2026-08-20)
- The 1024-query compliant SingleStream run was accidentally started **twice** (an armed
  chain raced my `pkill` by ~4 s and survived as an orphan, ppid=1, while a second copy
  started 15 min later). The two contended for the GPU and wrote the same log, so **their
  latency numbers are meaningless** — quarantined as `DISCARDED_contended_perf_full.log`.
- **What is unaffected:** the bounded 43-query reference run (04:33–04:39, sole GPU user)
  and therefore every E3 number currently reported; and the Choreo Dice values, which are
  accuracy, not timing.
- **Still to do:** one clean compliant run (~2 h on an idle GPU) to replace the
  `INVALID`-flagged bounded reference latency. `analyze_e3.py` already prefers
  `logs_perf_full/` automatically when present, so it only needs the run.
- **Lesson:** before launching on GB10, check for *orphaned* `run.py`/`main.py` (ppid=1),
  not just the wrapper scripts — see [[gb10-collection-gotchas]].

### E4 — REVIEW FINDINGS (2026-08-22): the bf16 test REFUTES the stated mechanism
Independent ASPLOS-style review caught errors I made. Verified independently — all confirmed.

**1. The bf16 manipulation does NOT confirm "decode is bandwidth-bound". It refutes it.**
NF4 -> bf16 raises weight bytes ~3.8x (2.1 -> 8.0 GB) but decode time only **1.68x**
(44.6 -> 75.2 ms/token). A bandwidth-bound phase given 3.8x the traffic returns ~3.8x the
time. Fitting `t = F + w*bytes`: **w = 5.18 ms/GB, F = 33.8 ms/token fixed**. So on GB10
decode is only **~24% weight streaming and ~76% fixed per-token framework cost** (HF eager
`generate` loop, no CUDA graphs, bnb dequant launches). I reported this test as
"independent confirmation of the mechanism"; it is the opposite.

**2. The genuinely good result hidden underneath.** The *marginal* bandwidth term is
device-invariant: **193 GB/s implied on GB10 vs ~191 GB/s on the M2** (~9% apart, two
independent instruments). That is a real mechanism-level finding and it is STRONGER than
the claim we made. Measured decode is 2.1x slower on GB10 not because of memory, but
because of a ~34 ms/token software constant vs ~9 ms/token on MLX.

**3. Instrumentation bug — the prefill/decode boundary is NOT comparable across backends.**
`stages/llm_huggingface/inference.py` stamps `first_token end` inside
`_FirstTokenCriteria.__call__` and the file contains **zero** `torch.cuda.synchronize()`
calls (verified). On an async CUDA queue that timestamp records when the CPU *enqueued*
work, not when the GPU finished it. MLX stamps at the first `stream_generate` yield, i.e.
after materialisation. **Prefill is under-counted and decode over-counted on CUDA** — the
exact direction of both headline asymmetries. The `run` total is safe (`batch_decode`
forces a sync); the SPLIT is not, and the split is the paper's whole claim.

**4. Decode compared per call, not per token (already fixed).** Backends emit different
token counts under greedy decoding, so per-call decode mixed speed with output length.
Per token the range is 0.47-1.89x, not 0.25-0.93x, and "never faster on GB10" was FALSE.

**5. The M2 leg is R=1 — which our OWN warm-up rule says to discard.** See
[[first-run-system-warmup]]: the first repetition of a cell is slower for its whole
duration, and dropping it flipped E2's sign. The entire mlx E4 dataset is run-1 data.
Also: arm order is collinear with a 7.5 h thermal soak, and the two devices ran at
different offered load (M2 rho ~0.17-0.20 vs GB10 ~0.42-0.61), so it is not a clean
device comparison.

**6. A cleaner flip already exists in the data and we did not claim it.** On GB10 alone,
factoid favours monolith_4b (3/3 runs) and multihop favours decomposed_shared (3/3 runs) —
a TASK flip with 3 replicates on both sides, far better supported than the device flip
(1 task, R=1, 1.17x margin). With 2 tasks x 2 devices, a device main effect cannot be
separated from a task x device interaction.

**7. Two of eight cross-device cells are truncated runs** (28/110 and 51/110 queries
completed, rho > 2) — and they produce the table's largest prefill speedups (15.00x,
11.07x) and its only decode ratios >= 1. Excluding them collapses prefill to 5.8-8.0x.

**8. `analyze_e4.py` mislabelled its count column** — it counts LLM calls, not queries
(fixed). No confidence intervals anywhere despite 3 cuda replicates. `n_prompt_tokens` is
never logged, so prefill cannot be normalised or roofline-checked at all.

### E4 — FIRST CROSS-DEVICE RESULT + the confound it raises (2026-08-20, r=1)
**The predicted asymmetry is there, and it is large.** mlx -> cuda speedup, per arm:

| arm | prefill speedup | decode speedup |
|---|--:|--:|
| factoid decomposed | **15.7x** | 1.05x |
| factoid decomposed_shared | 6.2x | 0.27x |
| factoid monolith (9B) | 8.1x | 0.62x |
| factoid monolith_4b | 6.2x | 0.42x |
| multihop decomposed | 11.1x | 0.64x |

Compute-bound **prefill speeds up 6-16x** on GB10; memory-bound **decode does not speed up
at all** (0.27-1.05x, i.e. often SLOWER on GB10). That asymmetry is precisely the mechanism
the flip claim rests on.

**Confound to settle before publishing this.** Both arms run 4-bit weights — mlx
`Qwen3.5-*-OptiQ-4bit`, cuda `quantize: true` = BitsAndBytes `load_in_4bit` (NF4) with
bf16 compute — so it is NOT a 4-bit vs bf16 mismatch (I checked `stages/llm_huggingface/
inference.py:131`). But **bitsandbytes NF4 is known to pay a per-token dequantisation cost
that hurts decode**, while MLX's 4-bit path is natively optimised. So "decode is
device-invariant" may partly be "bitsandbytes decode is slow", i.e. kernel quality rather
than memory bandwidth. A reviewer will raise this.
**RESOLVED 2026-08-20 — the confound is dead, and the test CONFIRMS the mechanism.**
Re-ran `factoid_monolith_4b_cuda` with `quantize: false` (bf16 weights, 4x the weight
traffic per token, no dequant), everything else identical:

| stage | metric | NF4 (4-bit) | bf16 | change |
|---|---|--:|--:|--:|
| Monolith LLM | prefill | 458 ms | 397 ms | **-13.3%** |
| Monolith LLM | decode | 1196 ms | 2029 ms | **+69.6%** |
| Monolith LLM | decode tok/s | 22.4 | 13.3 | -40.9% |
| Query rewrite | prefill | 317 ms | 279 ms | -12.3% |
| Query rewrite | decode | 619 ms | 878 ms | +41.8% |

Removing NF4 made decode **70% SLOWER**, not faster. Had the decode gap been a
bitsandbytes dequantisation artifact, dropping NF4 would have sped decode up. Instead the
two phases move in OPPOSITE directions under the same manipulation:
  * **decode slows with wider weights** -> weight-bandwidth-bound, as claimed;
  * **prefill speeds up without dequant** -> compute-bound, as claimed.
This is an independent manipulation confirming the prefill(compute)/decode(memory)
characterisation the flip claim rests on, and it answers the obvious reviewer objection
head-on. Artifact: `evaluation/self_rag/results/quantest/e4_quantest_bf16_cuda.csv`,
config `configs/quantest_factoid_monolith_4b_bf16_cuda.yml`.

### E4 — OPEN CONCERN found during collection (2026-08-20)
- **The serving config is SATURATED, so the phase split includes contention.** First
  `factoid_decomposed_mlx` run: 110 queries arrive at the pilot-tuned Poisson rate
  (0.1167 q/s) but only **28 reached the End stage** in 1917 s; per-role prefill share is
  **89–98%** and decode only **9–14 tok/s** (low for a 4B on an M2). With `queue_depth 110`
  several LLM stages run concurrently on one GPU, so each stage's wall-clock prefill/decode
  includes time-slicing against the others.
- **Why it matters:** E4's claim is about the prefill(compute)/decode(memory) BALANCE. A
  contended measurement inflates both phases and can distort their ratio, which is exactly
  the quantity the cross-device flip rests on.
- **Options:** (a) keep this as the *serving* view and report the caveat; (b) add a
  SERIALIZED pass (`serialize_queries`, `queue_depth 1`, as E3 uses) for a clean
  per-call phase characterisation, and use the serving runs only for end-to-end. (b) is
  the measurement that actually supports the flip claim. **Author decision needed.**
- **Early mlx signal (factoid, r=1) — two things worth keeping:** (i) *decomposition is
  prefill-amplifying* — prefill share 96.9% (decomposed) / 98.1% (decomposed_shared) vs
  **79.1%** (monolith 9B) / 83.8% (monolith 4B), because each sub-call re-reads the
  retrieved context but emits few tokens, converting decode work into repeated prefill;
  (ii) *decomposed* decodes at **11.5 tok/s** while *decomposed_shared* reaches **48.3** on
  the same 4B weights — consistent with four separate model copies causing memory pressure
  on the 16 GB M2, which the shared arm avoids. (ii) is a confound for the phase-balance
  claim on mlx and may not reproduce on the 128 GB GB10 — check the cross-device table.
- Note the rate knob was tuned before the e5-base-v2 retriever upgrade, so 0.6x-saturation
  may no longer hold — re-pilot the rate if we keep the serving config.

## E5 — Collocation (radt-orchestrated)
- **Thesis.** engine-specific contention; H2 phase-split control; per-engine attribution.
- **Done.** staged apparatus (configs, `analyze_staged.py`, `generate_stage_configs.py`, AMC sampler+
  calibration); R=1 sweep; thesis reframed.
- **Gap (Phase-0, signed off 2026-08-17).** extended dose ladder (B=4, stacked STREAM — **no ceiling** —
  + fg 0.7–0.8×); GB10 staged foreground → **bf16**; AMC calibration closure; verdict hygiene.
  **Thermal: log power/clocks only, NO gate yet** — check whether the earlier "thermal/scheduling"
  hand-wave is a real, persistent effect before adding any exclusion. + validator FAILs (p95 gate
  <500, blocked-puts); new tracing.
- **Code TODOs.** extend configs via `generate_stage_configs.py` (our `gen_configs.py`) — B=4 + stacked
  STREAM (uncapped) + fg 0.7–0.8×; GB10-**bf16** foreground configs; AMC calibration closure +
  agent→engine map; in `analyze_staged.py`: Fieller/cluster-bootstrap ratio CIs + no-verdict-at-R<2 +
  **log** power/clocks (diagnostic, no exclusion yet); fix p95-pool/blocked-puts.
- **Hyperparameters (pilot-tune).** per co-runner **R_max** (saturated rate, pilot per co-runner ×
  device) → intensities {0,25,50,75,100}%; foreground **λ** (near the bandwidth roof for the mechanism
  premise / 30–50% capacity for the B-sweep) + capacity fraction **0.7–0.8×** for the extended dose;
  **B** {0,1,2,4} + stacked STREAM (**uncapped**); decode length (256 tokens); warm-up (first 3 queries
  + first co-runner minute); **R=5**; foreground **precision** (**bf16 on GB10**); STREAM working-set
  size (≫ LLC); CLIP image batch; `queue_depth` (never-blocks); AMC calibration (agent→engine map,
  known-byte load magnitudes). *(Power/clocks are logged for thermal diagnosis — no gate/exclusion yet.)*
- **Data points.** AMC calibration: 1 known-load run/engine/device. Staged A–D (multi-pipeline configs,
  **radt-orchestrated via `main.py`**) × R=5 × `{mlx, cuda}` + the extended-dose cells.
- **Commands.**
  ```
  python evaluation/contention/generate_stage_configs.py            # writes the multi-pipeline YAMLs
  python evaluation/contention/amc_calibration.py --device mlx      # closure runs
  bash   evaluation/contention/collect.sh mlx  5                    # main.py <multi.yml> → radt schedules
  bash   evaluation/contention/collect.sh cuda 5
  python evaluation/contention/analyze_staged.py --device {mlx,cuda}
  ```

## Capacity sweep (stretch)
The Self-RAG monolith read along the size ladder (0.8B/2B/4B/9B/27B; 27B OOMs the M2). Configs exist
(`self_rag/configs/factoid_monolith_{0.8b,2b,27b}_*`); only 4B/9B have results. If time: add the three
rungs to E4's `collect.sh` (quality R=1 + resident memory), 27B as an expected-OOM datum.
**Hyperparameters (pilot-tune):** `n_quality` (120); resident-memory sampling cadence; R=1 (greedy).
*Fixed:* the ladder rungs {0.8B,2B,4B,9B,27B} (config `model` field only); quantization per device
(mlx 4-bit, cuda bf16).

---

## Light validation (replaces the heavy gate)
Fold a few sanity checks into each `analyze.py` (completed==expected, monotone perf timestamps, no
NaN/negatives, realized-rate near intended, ≥N pooled queries for a p95) and **warn** rather than
gate. No separate PASS/FAIL machinery.

## Sequencing
1. **P0** (tracing + harness) — gates everything.
2. **E1, E2** — fidelity; both devices (idle M2 + quiesced GB10); the corrected overhead numbers.
3. **E3** — MLPerf reference + Choreo on GB10 (parity, accuracy+perf) + Choreo 3D-UNet both devices
   (measurement boundary), R=5; ResNet cut.
4. **E4** — re-collect timing R=5 both devices + build the prefill/decode flip.
5. **E5** — Phase-0 fixes → AMC calibration → staged full-R (radt-orchestrated).
6. **Capacity (stretch)** if time.

## Decisions (2026-08-17) — all resolved
- **R:** E1, E2 = **R=10**; **E3, E4, E5 = R=5**. Quality cells R=1 (greedy).
- **E3 prong 1 = same-device parity on GB10:** run MLPerf's **reference harness** + Choreo on GB10 and
  match **accuracy AND performance** (same box → clean parity). Prong 2 (measurement boundary) uses
  both devices. **ResNet scenario-reduction: CUT.**
- **E5 Phase-0 signed off:** dose ladder B=4 + **stacked STREAM (no ceiling)** + fg 0.7–0.8×; GB10
  staged foreground → **bf16**; **AMC calibration** closure; **verdict hygiene**; **thermal = log
  power/clocks only, NO gate** (revisit only if throttling proves real).
- **Hosts:** idle M2 + quiesced GB10; **every experiment on both devices**; no agent sessions.

**Nothing left to decide.** The only remaining prerequisite is **P0** (deploy the new bulk+proc
tracing + build the minimal `gen_configs.py`/`collect.sh`/`analyze.py` harness). On your go, P0 is the
first thing implemented — then E1→E5 per the sequencing above.

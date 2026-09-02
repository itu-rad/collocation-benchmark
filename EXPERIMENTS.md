# Choreo — Experiment Definitions (source of truth)

**Authoritative, current description of the paper's experiments.** Where this disagrees with
`experimental_setup.tex` (the older 7-experiment draft) or any status snapshot, **this document
wins**. Companion: `EXPERIMENT_PLAN.md` (path to full collection). Last aligned: 2026-08-17.

**Five core experiments.** E1–E2 are instrument-fidelity; E3–E5 are application/capability studies
that escalate in workload complexity. E4→E5 chain (collocation reuses Self-RAG's foreground + its
prefill/decode split). A 6th, the **capacity/size sweep, is a stretch goal** (only if time).

| # | Name | Tier | Devices | One-line thesis |
|---|---|---|---|---|
| **E1** | NoOp overhead | fidelity | M2, GB10 | framework machinery (dispatch/queue/thread + zero-copy) is small, flat, O(1)-copy |
| **E2** | Modularity overhead | fidelity | M2, GB10 | wrapping *real* work (EfficientNet) in the graph ≪ the work itself; across model size × batch the **relative** overhead amortizes to ~0, absolute is ~fixed |
| **E3** | MLPerf / 3D-UNet | capability | GB10 (repro) · M2+GB10 (boundary) | we match MLPerf's **own reference harness** (accuracy + performance) on GB10, **and** show its offline-preload boundary hides the preprocessing online serving can't — a real, device-dependent share of per-request latency |
| **E4** | Self-RAG decomposition | application | M2, GB10 | decomposition cost = a prefill(compute)/decode(memory) rebalance whose optimum **flips across devices** |
| **E5** | Collocation | application | M2, GB10 | co-running CPU/GPU/ANE-bound work taxes the foreground **engine-specifically**, not as a fungible bytes/s tax |
| *(str)* | *Capacity sweep* | *stretch* | *M2, GB10* | *no-code size ladder (0.8B→27B); 16 GB M2 saturates (27B OOM), 128 GB GB10 has headroom* |

---

## E1 — NoOp / framework overhead
**Question.** What does Choreo's machinery cost, independent of any real work? Bounds per-stage
dispatch and the payload-passing model, so later overhead is attributable to the workload.
**Shows.** (1) per-stage dispatch small + **flat** vs depth (11.8 µs/stage on m2pro, 15.2 on gb10);
(2) context passing **zero-copy** by reference — flat at ~16 µs/stage from 0 to 10 MiB (0.033 µs/MB)
against a deep-copy counterfactual that grows with payload (115.0 µs/MB, reaching **74×** ref at
10 MiB). The `uninstrumented` / `+ tracing` configurations separate the tracing layer from the
framework.
**Artifacts.** `evaluation/overheads/framework_overhead/` — NoOp configs, `collect_e1.sh` +
`analyze_e1.py`, figures in `framework_overhead/paper_assets/` (each experiment owns its own
`paper_assets/`; there is no shared one).
**Status.** **Closed.** Re-collected on M2 Pro and GB10 (idle / X925-pinned), powers of two 1–128,
two configurations (`uninstrumented` / `+ tracing`), R=11 with run 1 dropped. Marginal per-stage
dispatch cost 11.78 µs (m2pro) / 15.18 µs (gb10). Zero-copy holds: reference passing is flat in
payload (0.033 µs/MB) against deep copy at 115.0 µs/MB. Unaffected by the 2026-08-28 polling fix —
verified by an interleaved same-session A/B on pinned GB10 (paired delta −1.55 µs at depth 1,
−22.98 at depth 10, both straddling zero).

## E2 — Modularity overhead (real workload, model × batch sweep)
**Question.** When the wrapped stage does *real* GPU work, is the decomposition overhead negligible,
and how does it behave as the work scales (model S/M/L, batch 1→64)?
**Shows.** EfficientNet fine-tune, monolith vs the same work as a Choreo pipeline: core wrapper
≈ **+49 µs/step (+0.13%)** at the canonical point; across the sweep the **relative** overhead (% of
step) shrinks toward zero as the step grows while the **absolute** cost (µs/step) stays ~fixed — a
small O(1) tax, not a scaling one. Measured with the *correct* (bulk+proc) tracing, the tracing
layer is also negligible. (The old "+4.5% in-process tracing" was a bug in the instrument, not a
property of the framework — see Infrastructure; we do not report it.)
**Artifacts.** `evaluation/overheads/modularity_overhead/` — `configs/torchvision_training.yml`
+ `gen_configs.py` (the Choreo side) and `baseline_finetune.py` (the monolith); `scale_sweep.yml`;
`collect_e2.sh` (three configurations: monolith / choreo / choreo-traced) + `analyze_e2.py`;
`modularity_overhead.md`.
**Status/gap.** Re-collecting on both machines (2026-08-27) against the current tracing, with time
per query as the metric of record — the earlier in-step metric sat below its own noise floor.

## E3 — MLPerf / 3D-UNet: reproduction + closing the measurement gap
**Question.** Two prongs: (1) **reproduce MLPerf's setup on our hardware** — run MLPerf's *own
reference harness* and Choreo's port of the same 3D-UNet/KiTS19 workload **both on GB10**, and show
Choreo matches the reference on **accuracy (Dice) AND performance (latency/throughput) on the same
device**. Because both run on GB10, performance parity is a clean apples-to-apples check that Choreo
faithfully reproduces the MLPerf experimental setup (not a bespoke harness). **GB10 only** — a
same-device faithfulness check, not a cross-device claim. (2) **the measurement boundary misleads
online** (both devices) — MLPerf preprocesses the
dataset **offline** (its QSL preload) and times only inference. That is valid for offline batch, but
in **online serving** a request arrives with its own raw data, so there is **nothing to preload** and
the preprocessing/loading sits **unavoidably on the per-request critical path**. Choreo measures true
end-to-end *per request* and reveals the preprocessing/loading portion MLPerf hides: it is
**variable across samples** and a **larger fraction on the faster device** (Amdahl: GB10's fast GPU
shrinks the inference denominator so preprocessing dominates more — inference ~95% of end-to-end on
M2 vs ~81% on GB10). The point is that this cost **cannot be hidden online** (you can't prefetch a
request you haven't received) — so it is real latency the standard's offline framing omits.
**Artifacts.** `evaluation/unet3d/` — one launch script (`collect_e3.sh`), one self-contained
analyzer (`analyze_e3.py`), the write-up (`unet3d.md`), four configs and `mlperf_reference/`. Stages
in `stages/unet3d_kits19/`, including `KiTS19DiceScore`, a port of the reference harness's
`get_dice_score` so the parity gate can be re-derived rather than cited from a log. Model: Zenodo
5597155 `.ptc`; data: `neheller/kits19` (both machine-local and gitignored). Prong 1 also needs the
**MLPerf reference harness** (`mlcommons/inference`, cloned under `scratchpad/`) run on GB10.

**Measurement.** Spans only — the `perf` configs set `disable_logs` on the pipeline and on all three
stages, so nothing is written per query. A separate `acc` config adds the scorer, runs R=1 and is
never timed, mirroring MLPerf's own AccuracyOnly/PerformanceOnly split; keeping output handling in
the timed graph would be the same error as the CSV write E2 removed. `KiTS19Preprocess` emits a
`case_size` marker span carrying `n_subvolumes` and the post-resample shape, so each case's size is
a property of its own trace instead of a join against another experiment's CSV. **E3 runs UNPINNED**
on both machines, unlike E1/E2: pinning throttles CPU preprocessing but not GPU inference, which
would inflate the very ratio this experiment reports, so `collect_e3.sh` refuses a `PIN`.

**Status/gap.** **CLOSED 2026-09-02.** R=6 per machine (repetition 1 dropped, 5 usable), zero
failures, 463 spans and 3 CSV rows per run, run times stable to 0.5%/0.7%. Prong 1 on GB10
against a `VALID` reference run: DICE **0.86329** vs 0.86168 (gate 0.85308), median inference
**5899 ms vs 5904 ms (−0.1%)**, matched per case median −0.2%. Prong 2: the hidden share is
**6.3%** on m3pro and **15.8%** on gb10 — 2.5x larger on the faster device, because inference
is 7.7x faster there while preprocessing is only 2.7x. Per case it ranges 1.7–27.1% and
8.5–68.4%; **more sub-volumes means a smaller share** (rho −0.38/−0.56), and because both
stages grow with n the share converges to a floor of 4.4%/7.8% rather than to zero. Known
caveat, measured and bounded: a device warm-up transient inflates the first 4 gb10 queries by
60–82%, is absent on m3pro, tracks run position rather than input shape, and leaves every
median unmoved. ResNet scenario-reduction: cut.

## E4 — Self-RAG decomposition & prefill/decode split (cross-device)
**Question.** What does decomposing agentic RAG cost vs a monolith, and *why*? The framework splits
each stage into **prefill (TTFT, compute-bound)** vs **decode (memory-bandwidth-bound)**.
**The point.** The DUTs have **similar memory bandwidth** but a **huge compute gap** (GB10 GPU ≫ M2
GPU; CPUs much closer). Since prefill is compute and decode is memory, the memory-bound decode is
~device-invariant while the compute-bound prefill is not → **the optimal prefill/decode balance, and
thus the decomposition recommendation, flips across devices.** A throughput-only serving benchmark
can't see this.
**Arms.** monolith-9B · monolith-4B · decomposed (4B, per role) · decomposed_shared (1×4B + mutex);
factoid + multihop; cuda (bf16) + mlx (4-bit); retriever `e5-base-v2` top_k=5; quality on a Haiku
judge (EM/F1 as guard).
**Artifacts.** `evaluation/self_rag/` — configs, `stage_latency.py`, `retry_analysis.py`,
`CHOREO_FINDINGS.md`, Haiku `judge/`; prefill/decode markers in the framework
(`log_first_token`/`log_generated_tokens`, `stages/stage.py`).
**Status/gap.** Quality matrix **validated** (16 cells both devices, greedy→exact, judge overturn=0).
**Missing:** the per-device prefill/decode **flip** figure (markers are recorded; the analysis isn't
built). Timing needs full-R with the new tracing (quality does not — greedy).

## E5 — Collocation (extension on Self-RAG)
**Question.** On a unified-memory SoC every engine (GPU/ANE/CPU) draws on one memory + its bandwidth.
Is an "idle" engine free, or does work on it tax the bandwidth-bound decode on the GPU — and **does it
depend on *which* engine** (CPU / GPU / ANE co-runner)?
**Thesis (reframed after the mock PC).** Contention is **engine-specific, not a fungible bytes/s tax.**
Negative control **H2 (phase split):** compute-bound prefill degrades far less than memory-bound
decode under the same co-runner — held vs the pure-bandwidth **STREAM** (CPU) co-runner, falsified vs
engine-sharing **CLIP**, same direction both devices. **H1:** decode tok/s vs co-runner GB/s.
**Design.** Foreground (RAG-serve / bare decode) + co-runner pipelines each in **its own process**,
sweeping the engine (CPU-stream / GPU-CLIP / ANE-CLIP) and intensity; staged single-diff A→B→C→D
(system → intensity → purify co-runner → purify foreground = the law). Orchestrated by **radt**
(multi-pipeline configs). MPS-off/on lever on GB10 separates scheduling- from resource-contention.
**Attribution.** M2 **counter-backed** — IOReport AMC per-requestor DRAM bytes
(`scripts/amc_bandwidth_sampler.{c,py}`). GB10 **proxy-backed** — `nvidia-smi utilization.memory`.
**Artifacts.** Design of record `CONTENTION_EXPERIMENTS_REDESIGN.md`; code `evaluation/contention/`
(`analyze_staged.py`, `staged_lib.py`, `generate_stage_configs.py`, `amc_calibration.py`, staged
configs); AMC sampler under `scripts/`.
**Status/gap.** R=1 done; thesis reframed. **Phase-0 apparatus fixes before full-R** (extended dose
ladder / B=4 + fg 0.7–0.8×, GB10 staged→bf16, AMC calibration closure, thermal gate, verdict hygiene).

---

## Cross-cutting infrastructure: tracing (NOT presented)

The framework's span export used to run on the workload's critical path (a bug: in-process
serialization holding the GIL → jitter; and the server dropped most traces). **This was infrastructure
we had to get right, not a paper contribution.** We fixed it with **bulk+proc** tracing (radt-owned
child process + one gzipped span artifact per run: negligible workload overhead, 100% capture).
**Consequence for the paper:** we present **none** of the old tracing numbers and **none** of the old
radt versions; **all experiments are recollected with the new tracing.** So E1/E2's "tracing overhead"
is simply "negligible," measured correctly.

- Choreo proc wiring: `utils/trace_span.py` + `main.py` hooks + span sites (uncommitted on
  `proto/proc-owned-tracing`). radt: **bulk** branch `itu-rad/radt @ feat/proc-owned-bulk-tracing`
  (to be tagged/pinned; supersedes the earlier proc-only PR fork). Runtime switch:
  `CHOREO_PROC_TRACE=1`. res17 experiment: **138** (real); **142** was throwaway.

---

## Execution model (see EXPERIMENT_PLAN.md for the concrete scripts)

Deliberately minimal:
- **configs/** — static, fully-explicit YAML, **one file per (variant × device)** (device,
  `serialize_queries`, loadgen, listeners all in the config — no CLI overrides). `gen_configs.py`
  writes them for sweeps (run once).
- **collect_<experiment>.sh** — a bash `for` loop over `configs/*.yml × runs` calling
  `python main.py <cfg> -e 138 --label <name>_r<r>`. **No `-p` → radt orchestrates** (single
  pipeline or the multi-pipeline collocation configs alike). `CHOREO_PROC_TRACE=1` set in the
  script; listeners on for E5. Each writes a timestamped log + summary and a provenance header
  (git commit, host, platform, library versions) under `collect_logs/`. The overhead experiments
  (E1, E2) are the exception on two counts: `-p 0` (no radt orchestration) and a local MLflow
  store instead of res17.
- **analyze_<experiment>.py** — reads `results/<machine>/` → matplotlib plots + plain tables.
  Self-contained: parsing, statistics, tables, LaTeX and figures in one file.
- **Removed:** `run_collection.py` (Cell/gate/marker machinery), `validate_pass.py`'s heavy gate,
  `generate_latex_results.py`. radt does all orchestration; we write no orchestration Python.

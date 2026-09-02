# Experiment Definitions (source of truth)

**Authoritative, current description of the paper's experiments.** Where this disagrees with
any older draft or status snapshot, **this document
wins**. Companion: `EXPERIMENT_PLAN.md` (path to full collection). Last aligned: 2026-08-17.

**Five core experiments.** E1–E2 are instrument-fidelity; E3–E5 are application/capability studies
that escalate in workload complexity. E4→E5 chain (collocation reuses Self-RAG's foreground + its
prefill/decode split). A 6th, the **capacity/size sweep, is a stretch goal** (only if time).

| # | Name | Tier | Devices | One-line thesis |
|---|---|---|---|---|
| **E1** | NoOp overhead | fidelity | M2, GB10 | framework machinery (dispatch/queue/thread + zero-copy) is small, flat, O(1)-copy |
| **E2** | Modularity overhead | fidelity | M2, GB10 | wrapping *real* work (EfficientNet) in the graph ≪ the work itself; across model size × batch the **relative** overhead amortizes to ~0, absolute is ~fixed |
| **E3** | MLPerf / 3D-UNet | capability | GB10 (repro) · M2+GB10 (boundary) | we match MLPerf's **own reference harness** (accuracy + performance) on GB10, **and** show its offline-preload boundary hides the preprocessing online serving can't — a real, device-dependent share of per-request latency |
| **E4** | Self-RAG execution strategies | case study | m3pro, gb10 | what decomposition *causes* — phase mix, lock serialization, resident weights, retry control flow — narrowed down per-stage/per-phase, and how the causal picture shifts across hardware |
| **E5** | Collocation types | case study | m3pro, gb10 | what the foreground actually contends on per collocation type, attributed per-pipeline; engine separation ≠ isolation on unified memory, counter-confirmed |
| *(str)* | *Capacity sweep* | *stretch* | *M2, GB10* | *no-code size ladder (0.8B→27B); 16 GB M2 saturates (27B OOM), 128 GB GB10 has headroom* |

---

## E1 — NoOp / framework overhead
**Question.** What does the framework's machinery cost, independent of any real work? Bounds per-stage
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
**Shows.** EfficientNet fine-tune, monolith vs the same work as a framework pipeline: core wrapper
≈ **+49 µs/step (+0.13%)** at the canonical point; across the sweep the **relative** overhead (% of
step) shrinks toward zero as the step grows while the **absolute** cost (µs/step) stays ~fixed — a
small O(1) tax, not a scaling one. Measured with the *correct* (bulk+proc) tracing, the tracing
layer is also negligible. (The old "+4.5% in-process tracing" was a bug in the instrument, not a
property of the framework — see Infrastructure; we do not report it.)
**Artifacts.** `evaluation/overheads/modularity_overhead/` — `configs/torchvision_training.yml`
+ `gen_configs.py` (the framework side) and `baseline_finetune.py` (the monolith); `scale_sweep.yml`;
`collect_e2.sh` (three configurations: monolith / choreo / choreo-traced) + `analyze_e2.py`;
`modularity_overhead.md`.
**Status/gap.** Re-collecting on both machines (2026-08-27) against the current tracing, with time
per query as the metric of record — the earlier in-step metric sat below its own noise floor.

## E3 — MLPerf / 3D-UNet: reproduction + closing the measurement gap
**Question.** Two prongs: (1) **reproduce MLPerf's setup on our hardware** — run MLPerf's *own
reference harness* and our port of the same 3D-UNet/KiTS19 workload **both on GB10**, and show
our implementation matches the reference on **accuracy (Dice) AND performance (latency/throughput) on the same
device**. Because both run on GB10, performance parity is a clean apples-to-apples check that the framework
faithfully reproduces the MLPerf experimental setup (not a bespoke harness). **GB10 only** — a
same-device faithfulness check, not a cross-device claim. (2) **the measurement boundary misleads
online** (both devices) — MLPerf preprocesses the
dataset **offline** (its QSL preload) and times only inference. That is valid for offline batch, but
in **online serving** a request arrives with its own raw data, so there is **nothing to preload** and
the preprocessing/loading sits **unavoidably on the per-request critical path**. the framework measures true
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

## E4 / §5.1 — Self-RAG execution strategies (case study)

**Register: investigative, not consumer.** A consumer benchmark outputs a verdict (this is faster).
This is a *causal account*: where the time and memory go, and which explanation survives isolation.
Rankings appear only as the observation that demands explanation, never as the deliverable.

**Question.** What does decomposing an agentic pipeline actually change — where do time and memory
go under each execution strategy, what *causes* the differences (phase mix, lock serialization,
resident weights, retry control flow), and does the causal picture itself shift across hardware?

**Arms — the four execution strategies, differing only in YAML.**
A monolithic prompt · B one model shared + locked (`depends_on_id`) · C per-role copies ·
D one model behind a server with continuous batching (`llm_server`). × {factoid, multihop} ×
{m3pro 24 GB MLX 4-bit, gb10 120 GB HF NF4}. Qwen3.5 4B/9B; 27B as the OOM-ceiling one-liner.
**Strategy D is Ollama-only and runs FIRST as a go/no-go** (configs exist, never run); vLLM is
dropped. If D fails, the text scopes to three strategies explicitly, never silently.

**Exhibits, in order** (question → manipulation → answer → mechanism):
1. Auto-generated pipeline graph + YAML-diff strip (13 stages, retry back-edge; the few declared
   lines separating the arms). Stage-reuse count across §4/§5 stated here, once. [C2]
2. The interplay table — quality × latency × throughput × peak memory × power/energy per strategy ×
   task × device, including the memory-budget answer (per-role copies comfortable at 120 GB,
   pressured/infeasible at 24 GB). Quality column non-negotiable. [C4]
3. Phase-breakdown figure **with a hardware track overlaid** — the unified profiler showing why.
   [C3, C4]

Calls-per-request variability on the cyclic graph reported alongside. **No predictive cost law**
and no unit-of-measurement critique — that register belongs to §4/E3.

**Data protocol.** Latency/quality from the existing **R=4 listener-off serial runs** (drop r1 →
R=3). Power/energy/memory from **new R=2 listener-on passes**; one paired on/off cell bounds the
observer cost. Poisson/throughput cells re-collected under the derived-λ rule.

**Status/gap.** Quality matrix validated (16 cells, judge overturn=0) but the **judge runner is not
committed** — it must be, for reproducibility. Serial R=4 exists both devices. **Missing:** strategy
D has never run; listeners are off on every serial config (so C3 has no data); peak memory unrecorded.

## E5 / §5.2 — Collocation types with per-pipeline attribution (build-up on §5.1)

**Question.** When two pipelines share the node, what is the foreground *actually contending on*
under each collocation type — the compute engine, the shared memory bandwidth, or nothing the
partitioning mechanism can reach — and can interference be attributed to the pipeline and resource
causing it? Degradation numbers are the symptom; the attribution is the deliverable.

**Subject: the collocation TYPE, not the background workload** (indexer / MemoryStream are props).
Capability shown: every number attributed to its pipeline — own process, own radt run, own listeners.
Foreground is the same Self-RAG serving pipeline as §5.1.

**Collocation axis.** m3pro — background on GPU / **ANE** / CPU (three engines, one unified-memory
pool). gb10 — time-sliced GPU / **MPS** / CPU. **MPS-only in the text**; MIG gets a half-day
verification and, as expected on this part, one line: "MIG is not supported on this device." radt
configures MPS automatically — answering by demonstration the gap the paper's own related work names
(manual MPS/MIG configuration expertise). [C3, C4]

**Exhibits.**
1. Per-pipeline attribution table — foreground p50/p95 + throughput per collocation type, background
   throughput alongside. [C2, C4]
2. **Separation ≠ isolation, counter-confirmed** (claim and proof merged): moving the background to
   ANE/CPU removes compute contention; degradation persists → unified-memory bandwidth is the shared
   resource. Proof by dose–response (MemoryStream at stepped intensities) with **AMC per-engine DRAM
   counters confirming the dose on m3pro**; on gb10 no DRAM counter exists (DCGM profiling dead on
   this stack) and the power/utilization proxy is stated in one sentence. [C3 — the flagship
   counter-explains-symptom exhibit]
3. What MPS partitioning buys vs time-slicing when the memory system stays shared. [C4]

**Grid.** Collocation types at one calibrated intensity everywhere; dose–response on m3pro only
(where the counters carry the narrative); gb10 types-only, overnight behind the occupancy gate, arm
order rotated.

**Artifacts.** `evaluation/contention/` (`analyze_staged.py`, `staged_lib.py`,
`generate_stage_configs.py`, `amc_calibration.py`); AMC sampler under `scripts/`; hardware facts in
`evaluation/contention/contention.md` (hardware-attribution facts consolidated there).

**Status/gap.** All §5.2 cells are new. Blocking: listeners on both devices, MPS path verified.

---

## Cross-cutting infrastructure: tracing (NOT presented)

The framework's span export used to run on the workload's critical path (a bug: in-process
serialization holding the GIL → jitter; and the server dropped most traces). **This was infrastructure
we had to get right, not a paper contribution.** We fixed it with **bulk+proc** tracing (radt-owned
child process + one gzipped span artifact per run: negligible workload overhead, 100% capture).
**Consequence for the paper:** we present **none** of the old tracing numbers and **none** of the old
radt versions; **all experiments are recollected with the new tracing.** So E1/E2's "tracing overhead"
is simply "negligible," measured correctly.

- framework proc wiring: `utils/trace_span.py` + `main.py` hooks + span sites (uncommitted on
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

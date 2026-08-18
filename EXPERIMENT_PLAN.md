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
  **analysis**. Retire `run_collection.py`, the `validate_pass.py` gate, and `generate_latex_results.py`.

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
- **Done.** Choreo config + `baseline_finetune.py` + `scale_sweep.yml` + `analyze_scale_panels.py`;
  CSVs on old tracing.
- **Gap.** re-collect with new tracing on both devices; port to the bash model. **The monolith arm is
  `baseline_finetune.py`, not `main.py`** — keep it as a second short loop (it is the non-framework
  reference; that's fine).
- **Hyperparameters (pilot-tune).** `max_batches`/run + warm-up steps dropped (currently 1100 / first
  200 — re-verify steady-state flatness); **R=10** (decided — cheap fidelity cell); the canonical
  batch anchor (b8). *Fixed:* batch
  {1,2,4,8,16,32,64}, model {S,M,L}, fine-tune head; **`num_workers=0`** — a control (not tuned) that
  removes the concurrent-prefetch data-path confound so the per-step metric isolates the framework
  wrapper, identically in both arms.
- **Code TODOs.** `gen_configs.py` (batch {1,2,4,8,16,32,64}@S + model {S,M,L}@b8, × device →
  static YAMLs); `collect.sh` (Choreo arm via `main.py`, tracing on **and** a tracing-off arm via
  `CHOREO_DISABLE_TRACING=1`, + the baseline loop); reuse `analyze_scale_panels.py` as `analyze.py`.
- **Data points.** `{baseline, choreo core, choreo traced}` × sweep cells × `{mps, cuda}` × **R=10**.
- **Commands.**
  ```
  python evaluation/overheads/modularity_overhead/gen_configs.py
  bash   evaluation/overheads/modularity_overhead/collect.sh mps 10
  bash   evaluation/overheads/modularity_overhead/collect.sh cuda 10
  python evaluation/overheads/modularity_overhead/analyze_scale_panels.py --device {mps,cuda} --fig e2.png
  ```

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

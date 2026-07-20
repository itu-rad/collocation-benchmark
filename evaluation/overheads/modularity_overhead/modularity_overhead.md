# Modularity overhead — EfficientNetV2-S, monolith vs Choreo

**Question.** Empty stages being cheap (the NoOp microbenchmark, `../framework_overhead/`)
does not prove that wrapping a *real* workload in Choreo's graph/queue/thread
structure is cheap. Does expressing an EfficientNetV2-S Imagenette fine-tune as a
Choreo pipeline inflate per-step latency versus a hand-written, monolithic PyTorch
implementation? This is the real-workload, vs-an-external-baseline counterpart to NoOp.

**Answer (DGX Spark cuda):** the framework's **core wrapper adds +49 µs/step (+0.13 %)** —
a tiny, fixed, negligible positive overhead. With **tracing enabled** (the `async_tracing`
radt branch) it adds **+1.75 ms/step (+4.5 %)** — modest on a fast 39 ms vision step,
negligible on the seconds-scale stages of the case studies. Modularity is cheap.

---

## Setup

- **Workload:** transfer-learning fine-tune of **EfficientNetV2-S** on Imagenette —
  frozen backbone, replaced 10-class head, Adam (lr 1e-3), cross-entropy, **batch 8**.
  A deliberately *newer* model that is still far smaller than the transformer workloads
  now common, so the per-step compute is a realistic mid-range target.
- **Two implementations, identical work:** (a) `baseline_finetune.py` — standalone PyTorch,
  no framework, no tracing (`--no-radt`); (b) the Choreo pipeline
  `configs/torchvision_training.yml` (TorchVisionDataLoader → TorchVisionClassification).
  Both share the same data loading, the same frozen backbone, the same Adam/CE step, the
  same **12,810** trainable params, and both call `torch.{cuda,mps}.synchronize()` at step
  end. Both run the **train split only**, `num_workers=0` (see "Isolating the wrapper").
- **radt:** the `feat/Sipondo/async_tracing` branch (commit 3ba61cb), which exports MLflow
  trace spans **asynchronously** instead of blocking at span close. This is what makes the
  tracing-on number realistic; the pip 0.2.28 sync export inflated it ~8× (see Result 2).
- **Devices (both DUTs):** DGX Spark **cuda** (GB10, torch 2.11+cu130, driver 580.95.05) —
  reported below — and M2 Pro **mps** (user runs the same driver `--device mps`). Conclusions
  are about *direction within a device*, never cuda-vs-mps µs.
- **Clock:** monotonic `perf_counter_ns` (trailing CSV column); wall-clock col 0 is kept only
  for RadT alignment.
- **Scheduler:** closed-loop `OfflineLoadScheduler`, one query (one batch = one step) in flight
  (`serialize_queries: true`).
- **Steady-state protocol:** each run is **one continuous epoch (1,100 steps)**; the first
  **200** are dropped (cuDNN autotune / first-call) and the rest pooled across **R = 5** runs
  → 4,500 steps/arm. Per-step latency is **flat within a run** (verified — see below), so runs
  need no cooldown; the driver **interleaves** the arms so all see the same conditions. Median
  + two-sample bootstrap 95 % CI.
- **Two arms** (as in NoOp): **tracing off** (`CHOREO_DISABLE_TRACING=1`, core dispatch) and
  **tracing on** (the MLflow span layer). The baseline has no tracing.

---

## Metric definitions (monotonic perf column)

- **Per-step training latency** — baseline `training_step` start→end vs Choreo
  `EfficientNet training` start→end. Both bracket the *same* GPU work
  (`.to`→zero_grad→forward→backward→step→**synchronize**) and both **exclude** data loading.
  One-in-flight ⇒ the stage's start/end rows strictly alternate ⇒ consecutive pairing is exact.
- **Overhead** = (median_Choreo − median_baseline), in **absolute µs** *and* as a ratio %, each
  with a **two-independent-sample** bootstrap CI (the arms are separate process runs, not paired).

---

## Result 1 — Modularity adds a negligible, fixed overhead (core dispatch, tracing off)

Per-step training latency at steady state (cuda, 4,500 steps/arm):

| arm | median (ms) | 95% CI |
|---|---:|:---:|
| baseline (monolith) | 39.171 | [39.144, 39.195] |
| Choreo (tracing off) | 39.221 | [39.190, 39.252] |

**Overhead: +49.1 µs [+11.4, +90.9]; +0.13 % [+0.03 %, +0.23 %].** A small, consistent
**positive** cost — **~0.1 % of a 39 ms step**, i.e. negligible. This is the framework's true
per-step wrapper cost (queue hand-off + stage log + the `_push_to_outputs`/span machinery that
falls inside Choreo's measured bracket). It is radt-version-independent (tracing is disabled in
this arm). It is the expected *direction* (the wrapper can only add work) and the expected
*magnitude*: five independent code-audit agents found the Choreo bracket carries ~30–150 µs of
extra work (`queue.put` + `mlflow.start_span` enter/exit after `synchronize()`), and +49 µs
lands squarely in that range. (`analyze_operational_overhead.py --device cuda`.)

### Why an earlier run *looked* like Choreo was 2 % faster (a measurement artifact, now fixed)

A first collection — all five baseline runs first, `num_workers=2`, 400-step runs — produced a
*negative* overhead (Choreo −2.1 %), i.e. the framework looking **faster** than the code it
wraps. A framework beating its own baseline is a red flag, so we tracked it down rather than
report it. Two confounds, neither of them framework behavior:

1. **Run scheduling + DataLoader workers.** The per-run medians showed the baseline degrading
   while Choreo stayed flat:
   ```
   baseline   r1..r5 = 40.45 40.87 41.36 42.04 42.41   (num_workers=2, all run first)
   choreo_t0  r1..r5 = 40.53 40.25 40.71 40.73 40.63   (flat)
   ```
   A 3,000-step **continuous** probe proved the per-step time is **flat within a run** (39.1 ms,
   zero drift) — so this is *not* within-run GPU thermal throttling. The cross-run climb tracked
   `num_workers=2` worker-process effects under dense back-to-back baseline scheduling; with
   `num_workers=0` and continuous runs the baseline is flat.
2. **DataLoader prefetch contention.** `num_workers=2` decodes the *next* batch concurrently
   with the step (~7 % slower than `num_workers=0`); serialized Choreo's loader is idle during
   the step, so it avoided the tax — an execution-model difference, not wrapper overhead.

**Fix (now the `run_modularity.py` defaults):** `num_workers=0` in both arms (removes prefetch
contention and any data-path asymmetry), single continuous epoch per run at steady state, and
interleaved arms. At that matched operating point the artifact is gone and the true +0.13 %
wrapper cost emerges. (GB10 clock-pinning would also help but needs root.)

## Result 2 — Overhead-in-context (the tracing layer)

The real EfficientNetV2-S step is **39.17 ms**. Per-step framework cost as a fraction:

| layer | per-step cost | % of this step |
|---|---:|---:|
| core dispatch (Choreo off − baseline) | +49 µs | +0.13 % |
| MLflow tracing layer (on − off), async radt | +1.75 ms | +4.5 % |

Core dispatch is negligible. The **tracing layer** (MLflow spans via the `async_tracing` radt)
is a *fixed* +1.75 ms/step — **+4.5 % of this fast 39 ms vision step, and <0.1 % of a
seconds-scale LLM or retrieval stage** (the case studies). This is why tracing is a separable,
optional arm: effectively free where stages are expensive, and tunable for very fast steps.
§exp-noop shows the per-stage cost is fixed and independent of step size, so its fraction
shrinks as stages grow.

> **The radt branch matters here.** With the pip `radt==0.2.28` (synchronous span export that
> blocks at span close), this same arm measured **+14 ms (+36 %)** — an ~8× inflation that is an
> artifact of blocking export, not the cost of tracing. The `async_tracing` branch exports off
> the critical path; the median drops to +1.75 ms. (A tail remains — p95 ≈ 102 ms — from periodic
> async-queue flushes; the median is the right summary of the per-step cost.) These numbers are
> still a local-store export on the `-p 0` path; an orchestrated run against RadT's server may
> differ, but the async mechanism is now the real one. (`true_overhead_analysis.py --device cuda`.)

## Result 3 — End-to-end breakdown (descriptive)

| component | median (ms) [95% CI] |
|---|---|
| baseline training step | 39.171 [39.144, 39.195] |
| Choreo training stage | 39.221 [39.190, 39.252] |
| Choreo dataloader stage | 33.85 [32.75, 35.15] |
| Choreo end-to-end / query | 74.47 [73.38, 75.71] |

With `num_workers=0` the dataloader stage is a synchronous JPEG decode of 8 full-res images
(~34 ms) — measured as its own first-class stage, end-to-end visibility a monolith hides. Under
serialized one-in-flight execution it does **not** overlap the training step, so end-to-end per
query (74.5 ms) ≈ decode + train. (Cross-query pipelining — overlapping decode with training —
is a separate framework capability, deliberately *off* here so the per-step metric is clean.)
(`breakdown_overhead.py --device cuda`.)

---

## Mechanism

Both per-step brackets time identical GPU work and exclude data loading; the compute is
bit-for-bit the same workload. Choreo's measured bracket additionally contains the stage
start/end log, a `queue.put` (`_push_to_outputs`), and the `mlflow.start_span` enter/exit —
all *after* `synchronize()`. That CPU work is the +49 µs. With tracing on, the MLflow span
machinery (attribute construction + handing the span to the async export queue) adds the
+1.75 ms. Because the EfficientNetV2-S step is ~39 ms of GPU compute, the core wrapper is a
0.1 % tax and the tracing layer a few percent.

## Caveats / threats to validity

- **The +0.13 % is significant but negligible.** The CI excludes 0 (it is a real fixed cost,
  not noise), but it is 0.1 % of one step — we report it as the wrapper's true cost, not as
  "zero". The earlier −2 % was an artifact (above), not a framework speedup.
- **`num_workers=0` is deliberate** — it removes both the prefetch-contention confound and the
  decode-vs-launch thread asymmetry, isolating the wrapper. It makes data loading slow
  (Result 3), but that is outside the per-step training metric.
- **Tracing-on uses the `async_tracing` radt + a local store.** The `-p 0` path has no RadT
  MLflow server, so spans export async to an isolated local file store; the case studies'
  as-used server cost may differ. The async export also leaves a p95 tail (periodic flushes);
  the median is the per-step summary.
- **Single batch size / model.** EfficientNetV2-S batch 8 only; the step-size-independence of
  the overhead is carried by §exp-noop rather than a stress config here.
- **Cross-DUT.** cuda and mps µs are not comparable; we report per device, direction only.
  The M2 Pro mps half is collected by the user with the same driver and radt branch.

## Reproduce

```bash
# radt: the async_tracing branch (env specs install it; or editable):
#   pip install -e <radt-checkout>/radt   # feat/Sipondo/async_tracing
# collect (inside the torch+cuda env), per device
python evaluation/overheads/modularity_overhead/run_modularity.py --device cuda --runs 5
python evaluation/overheads/modularity_overhead/run_modularity.py --device mps  --runs 5   # M2 Pro
# (defaults: --num-workers 0, --cooldown 0, --max-batches 1100, interleaved, both arms)

# analyze (--warmup 200 = steady-state cutoff)
python evaluation/overheads/modularity_overhead/analyze_operational_overhead.py --device cuda
python evaluation/overheads/modularity_overhead/breakdown_overhead.py           --device cuda
python evaluation/overheads/modularity_overhead/true_overhead_analysis.py        --device cuda
python evaluation/overheads/modularity_overhead/generate_latex_results.py        --device cuda > table2.tex
```
CSVs: `results/mod_{baseline,choreo_t{0,1}}_d{cuda,mps}_r{R}.csv`; env in `run_modularity_env.txt`.

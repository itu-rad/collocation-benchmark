# E2 — the cost of decomposition: EfficientNetV2 monolith vs Choreo

**Question.** Empty stages being cheap (the NoOp microbenchmark, `../framework_overhead/`)
does not prove that wrapping a *real* workload in Choreo's graph/queue/thread structure is
cheap. Does expressing an EfficientNetV2 Imagenette fine-tune as a Choreo pipeline cost
throughput against a hand-written, monolithic PyTorch implementation — and where does the
cost land? This is the real-workload counterpart to E1, and the one measured against an
external baseline rather than against the framework's own configurations.

**Status.** Re-collecting on both machines since 2026-08-27. The apparatus, the metric and
the statistics below are settled; the numbers are not yet in. See *Results* at the end.

---

## What is run

Three things per cell, on an identical workload — the `(model, weights, batch)` triple is
read out of the same YAML and handed to both sides, so the only difference is the framework:

| configuration | what it is | how |
|---|---|---|
| `monolith` | a bare PyTorch loop, no framework at all | `baseline_finetune.py` |
| `choreo` | the same workload declared as a Choreo pipeline | `main.py`, `CHOREO_DISABLE_TRACING=1` |
| `choreo-traced` | the same pipeline with radt proc tracing on | `main.py`, `CHOREO_PROC_TRACE=1` |

    cost of decomposition = choreo        − monolith
    cost of tracing       = choreo-traced − choreo

Their order is rotated by repetition index so none of them always absorbs the warm-up. A
fixed order systematically penalises whichever runs first, which is where an earlier cell's
impossible reading of "the wrapper makes work faster" came from.

**Workload.** Transfer-learning fine-tune of EfficientNetV2 on Imagenette — frozen backbone,
replaced 10-class head, Adam (lr 1e-3), cross-entropy. One query = one batch = one training
step. Both sides share the data loading, the frozen backbone, the Adam/CE step, the same
trainable parameter count, and both synchronise the accelerator at step end. Train split
only.

**Sweep.** EfficientNetV2-S at batch {1, 2, 4, 8, 16, 32, 64}, plus EfficientNetV2-M and -L
at batch 8 — 9 cells, on each of `m2pro` (Apple M2 Pro, torch `mps`) and `gb10` (DGX Spark,
torch `cuda`). ConvNeXt-L was dropped. Conclusions are about direction *within* a machine,
never m2pro-vs-gb10 microseconds.

---

## The metric of record: time per query

**Time per query** — start-to-start between consecutive queries, equivalently time per batch,
equivalently 1/throughput.

Two properties make it the right one here:

- **It is anchor-invariant.** In steady state the same period comes out whether you cut at
  the pipeline row, the training row, or the monolith's own step row: measured on a b8 cell,
  66118 µs from pipeline starts vs 66242 µs from training starts. That is what makes two
  processes comparable although they emit *different markers* — which is exactly the
  constraint `disable_logs: true` imposes on the Choreo side.
- **It covers the whole cycle**, data loading and preprocessing included, where an in-step
  marker is blind to the 10–30 % spent there.

**What it replaced, and why.** The previous metric was the training step's own duration,
compared across the two processes. It is not usable at this noise floor: the framework's
cost lands mostly *between* steps, which that marker excludes by construction. It measured a
near-zero difference against ±600 µs of run-to-run noise and flipped sign between
repetitions — per-run paired differences for one cell ran
`+7.6 / +30.0 / −50.6 / +21.5 / −6.0 / −1241.8 / −55.2 / +8.1 / −18.6`, and one cell settled
at −575 µs, i.e. "the wrapper makes work faster". The honest reading of those tables was
"in-step overhead is below a ±600 µs noise floor", not the signed values printed.

## Co-headline: the query latency breakdown

From the spans of the `choreo-traced` runs: per-stage latency (**dataloader**, **training**)
plus the auxiliary framework overheads (**entry**, **handoff**, **exit**, **turnaround**).

These are successive instants within *one* query on *one* clock, so unlike any cross-process
difference they are non-negative by construction and carry no run-level term. They sum to
the time per query exactly — E1 verified the identity at a residual of **0.000 µs over 300
queries** on a 2-stage pipeline. Negative intervals are therefore a hard failure of the
analyzer, never something to take a median over. (Stage CSV rows carry no query id, so an
index-aligned pairing silently shifts every subsequent step when one row is dropped; that
had already produced a transition sum of −63842 µs.)

Published alongside it is the identity that ties the two headlines together:

    Δ(time per query) = (dataloader_choreo − gap_monolith)
                      + (training_choreo   − step_monolith)
                      + Σ transitions

which can show that part of what looks like added "transition" cost is work that *moved out
of the stage bodies* rather than work added. Suppressing that term would be the difference
between an honest decomposition and a flattering one.

## Keeping the instrument out of the measurement

E1's finding was that a log row is not free — 7.7 µs single-threaded, far more under
stage-thread contention on the logging handler lock — so an experiment that measures the
framework through its own CSV logger measures the logger too.

- Both Choreo configurations run **`disable_logs: true`**, so neither writes per-stage CSV
  rows and no synchronous write+flush sits inside the measured interval on one side only.
  Time per query survives this because it comes from the pipeline-level rows, which
  `pipeline.py` emits unconditionally (`disable_logs` is a Stage flag; Pipeline has none).
  What is given up is the per-stage breakdown, which the spans give back at a fraction of
  the per-event cost.
- The monolith writes through **the same synchronous `FileHandler` that `main.py` installs**.
  It previously used a `QueueHandler`/`QueueListener` *and* a stderr sink, so the two sides
  differed in three ways at once — rows per step, async vs synchronous emit, and a sink whose
  cost depends on whether stderr is a tty. Matching them costs the monolith ~+15 µs/step
  against effects of 240–2100 µs, and removes its listener thread, making the control an
  actual single-threaded loop.
- **`num_workers = 0`** on both sides — a control, not a tuned knob. It removes the
  concurrent-prefetch data path so the metric isolates the wrapper. There is no preloading
  or prefetching anywhere in this experiment.
- **Clock:** monotonic `perf_counter_ns` (the trailing CSV field). Wall-clock column 0 is
  kept only for radt alignment and is never used for timing.
- **Store:** a local MLflow store, not res17. This is the documented exemption for the
  overhead experiments (E1 and E2) only — they emit far more spans per second than any real
  workload, and that volume is not what the remote server is there to carry.

## Statistics

- **300 steps per run, first 50 dropped.** Step time is flat from step 0 on the GB10
  (38.8, 38.7, 38.7 … ms across 900 steps) and flat on a warm M2 Pro run, so the old
  1000/200 was never buying within-run settling.
- **R = 11 collected, run 1 dropped → 10 usable, and the drop is the default.** The first
  repetition of a cell is measurably slower for its *whole* duration — an anchor measured
  97.2 ms/step for r1 against 89.2 for r2..r6, still 97.4 ms in its *last* 300 steps — so
  per-step warm-up dropping cannot remove it. On the E2 smoke collection, leaving it in moved
  one breakdown term by 800 µs.
- **Statistic of record: the paired across-run difference.** The configurations are
  interleaved within each repetition, so runs pair by id:
  `d_i = median(choreo_i) − median(monolith_i)`. Combined across runs by the **median**
  (robust to one contaminated repetition). The CI resamples run **pairs** with replacement
  and re-resamples queries within each chosen run — the run is the unit of replication.
- Per-run paired differences are printed beside every interval, so a single bad repetition
  is visible rather than absorbed.
- A **negative cost is not a speed-up**: it means the difference is smaller than what this
  apparatus resolves at that cell. It is reported as measured rather than clipped.

## A span whose count is wait-dependent

`pipeline.py`'s result loop spans **every poll**, including the ones that time out empty
after 0.1 s. Its count is therefore proportional to how long the pipeline spends waiting,
not to how many queries ran: on EfficientNetV2-L b8 the same 300 queries produce ~3006 spans
on gb10 (243 ms/query → ~4 polls each) and ~3890 on m2pro (570 ms/query → ~7 polls each),
against a fixed 6 per query. Two consequences:

- **The total span count is not a deploy-integrity check here.** In E1 a span-count mismatch
  is what caught a stale deploy; in E2 the totals legitimately differ across machines and
  drift between runs. Check the six fixed per-query types instead, and check code identity by
  checksum.
- **It lands in the cost of tracing, and it scales with waiting rather than with work.** Each
  poll span is export work charged to the `choreo-traced` configuration, so a slower machine
  appears to pay more tracing cost per query for a reason that has nothing to do with the
  query. `choreo − monolith` is unaffected, and so is the breakdown (which is keyed by query
  id and uses only the six fixed types); only the tracing number carries this component, and
  it should be reported as what tracing costs *in this configuration* rather than as a
  per-query constant.

## Caveats to state in the paper

- `serialize_queries: true` means E2 measures the framework with **pipelining disabled** —
  the configuration where it can only lose.
- Choreo runs **6 threads to the monolith's 1**. That is genuine modularity cost, and it
  should be named rather than left implicit.
- The `-p 0` and radt-orchestrated paths install *different* logging instruments
  (`main.py` vs `utils/logger.py`). Nothing may be compared across them.

---

## Reproduce

```bash
python evaluation/overheads/modularity_overhead/gen_configs.py     # 9 cells x 2 devices
bash   evaluation/overheads/modularity_overhead/collect_e2.sh m2pro 11
bash   evaluation/overheads/modularity_overhead/collect_e2.sh gb10  11   # taskset-pinned
python evaluation/overheads/modularity_overhead/analyze_e2.py
python evaluation/overheads/modularity_overhead/analyze_e2.py --latex gb10 > table2.tex
```

CSVs land in `results/mod_<cell>_<configuration>_<machine>_r<N>.csv`; each collection writes
a timestamped log and summary TSV to `collect_logs/`, headed by a provenance block (git
commit and dirty flag, host, platform, python/torch/radt/mlflow versions, pinning, run and
step counts). Figures go to `paper_assets/`.

---

## Results

**Pending the 2026-08-27 collection.** Numbers, tables and figures go here once
`analyze_e2.py` runs on the complete dataset; they are deliberately not carried over from
the previous collection, every part of which was superseded — different metric, different
logging instrument on the monolith side, fixed configuration order, and pre-dating the
`get_input` span removal, the marker spans and the cyclic-pipeline termination fix.

Gates that must hold before any number here is reported:

- every cell has 11 repetitions per configuration, zero failures, no truncated CSVs —
  verified from the CSVs on disk, not from the log;
- the **fixed** per-query span types number exactly six per query on both machines
  (`pipeline query`, `pipeline query processed`, and `.run` + `.push_to_outputs` for each of
  the two stages) — see the note below on why the *total* span count cannot be used for this;
- no negative transition intervals anywhere;
- the Δ(time per query) identity closes to within a stated tolerance;
- no cell reports a negative cost of decomposition without an explanation.

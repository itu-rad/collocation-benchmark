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

Collected 2026-08-28 on both machines against commit `d242d80`: 9 cells x 3 configurations
x 11 repetitions = 297 runs each, zero failures, verified from the CSVs on disk.

**These numbers are from the second collection.** The first (2026-08-27) was discarded after
it showed that the framework's own result-collector and drain loop were waking ten times a
second for the whole duration of every query, which cost measurable time inside the
dataloader stage. That is fixed (`pipeline.py`, condition variables in place of 100 ms
polling); see *A polling artifact, found and removed* below, because the correction changes
what the breakdown says.

### The cost of decomposition

**GB10** (pinned to the X925 cluster; the quiet machine, and the one to read):

| cell | time per query | cost of decomposition | as % |
|---|--:|--:|--:|
| EfficientNetV2-S b1 | 13.81 ms | +1859 µs [+1332, +2149] | **+13.46%** |
| EfficientNetV2-S b2 | 19.58 | +1465 [+930, +1770] | +7.48% |
| EfficientNetV2-S b4 | 34.02 | +1498 [+1141, +1774] | +4.40% |
| EfficientNetV2-S b8 | 63.91 | +1548 [+1061, +2202] | +2.42% |
| EfficientNetV2-S b16 | 128.94 | +1437 [+962, +1826] | +1.11% |
| EfficientNetV2-S b32 | 268.63 | +1433 [+701, +2112] | +0.53% |
| EfficientNetV2-S b64 | 554.79 | +2404 [+398, +3536] | **+0.43%** |
| EfficientNetV2-M b8 | 134.52 | +476 [+130, +1029] | +0.35% |
| EfficientNetV2-L b8 | 225.22 | +1419 [+948, +1794] | +0.63% |

Two things at once, which is the result:

- **The absolute cost is flat.** 1433-2404 µs across a **40x** range of query time. It does
  not scale with batch, with model, or with payload.
- **The relative cost therefore amortizes**, 13.46% -> 0.43%. Across models at fixed batch,
  2.42% -> 0.63%.

A fixed per-query tax, not a scaling one. That is the claim E1 makes on empty stages,
reproduced here on a real workload against an external baseline.

### Where the cost goes

From the spans of the traced configuration, GB10 (µs per query):

| cell | entry | handoff | exit | turnaround | **framework** | framework % |
|---|--:|--:|--:|--:|--:|--:|
| S b1 | 170.4 | 105.1 | 768.9 | 525.5 | **1569.8** | 10.23% |
| S b8 | 179.5 | 164.7 | 457.6 | 852.5 | **1654.3** | 2.53% |
| S b64 | 182.7 | 201.7 | 801.3 | 684.5 | **1870.3** | 0.34% |
| M b8 | 182.4 | 188.4 | 595.5 | 758.0 | **1724.3** | 1.28% |
| L b8 | 179.2 | 185.0 | 745.1 | 651.2 | **1760.5** | 0.78% |

`entry` is flat to within 25 µs across the whole sweep. **`handoff` runs 105 -> 202 µs while
the payload grows 64x** — E1's zero-copy result reappearing in a real workload, and the
reason the absolute cost does not scale. Full nine-cell tables are in the analyzer output;
the figure is `paper_assets/e2_query_latency_breakdown.png`.

### What tracing costs

Within noise at **all nine** GB10 cells: every interval straddles zero, from -245 µs to
+289 µs against a query of 14-555 ms. Turning tracing on is not measurable at this workload
scale, which is what licenses using the traced configuration to source the breakdown.

### The identity, and its tolerance

`Δ(time per query) = (dataloader - monolith gap) + (training - monolith step) + framework`

closes on GB10 with residuals of **+74 to +1321 µs** (median +380), against measured costs of
476-2404 µs. Two known contributors, both stated rather than absorbed:

1. **The two sides use different configurations.** The breakdown comes from `choreo-traced`
   (spans only exist when tracing is on) while `measured` is `choreo - monolith`, untraced.
   They differ by exactly the cost of tracing — bounded above, but not zero.
2. **Median of sums is not the sum of medians.** Each term is a median over runs.

`dl - gap` is now negative at every GB10 cell (-63 to -1382 µs): the Choreo dataloader is no
slower than the monolith's own inter-step loading. Before the polling fix it reached
**+2984 µs** at b64 and grew with batch, which is what prompted the investigation.

### The M2 Pro half is weaker, and should be reported as such

Same apparatus, same code, but **5 of 9 cells have a confidence interval that straddles
zero**, and three of those read negative (-650, -736, -166 µs). The machine is unpinned and shares itself with the OS; its
run-to-run spread swamps a sub-1% effect. A negative cost is not a speed-up — it means the
difference is below what this apparatus resolves on this machine.

The *within-query* breakdown survives, because it never crosses a process boundary:
across the batch sweep b1 -> b64, framework overhead 570 -> 2534 µs and its share
1.61% -> 0.26% (the lowest share of any cell is 0.18%, at EfficientNetV2-L b8); `entry` is
flat at 63-76 µs and `handoff` at 74-138 µs across the same 64x payload range.

**One term does not behave: `exit`, and it is now explained and fixable.** It grows
248 -> 2081 µs with batch on the M2 Pro while GB10's stays in the 425-801 range. `exit` is
the interval from the training stage's `push_to_outputs` to `pipeline query processed` — the
finished query crossing from the training thread to the collector thread.

**Cause: returning the batch's resident pages to the OS, on the collector thread.** The
training stage returns the same query object it received and never clears `query.data`
(`stages/torchvision_classification/classification.py`), so the input batch — 113 MB at
S b64 — rides through to the collector, where the query goes out of scope and the memory is
released. Note `inputs = inputs.to(self._device)` rebinds a *local*, so what is released is
the **host** tensor, not the device copy.

Freeing that memory is nearly free on Linux and expensive on macOS:

| payload | free on m2pro | free on gb10 |
|--:|--:|--:|
| 1.8 MB | 37.8 µs | 1.8 µs |
| 56.6 MB | 607.7 µs | 0.7 µs |
| 113.2 MB | **1223.6 µs** | **0.7 µs** |

macOS returns dirty pages to the kernel synchronously on free; glibc retains them in its
arena. That ~1750x difference at 113 MB is the whole of the machine asymmetry — it is a
platform allocator property, not a framework-logic one.

**Confirmed by intervention, not just correlation.** Clearing `query.data` before the
hand-off (the payload is dead by then; nothing downstream reads it) on an otherwise identical
b64 traced run:

| | `exit` |
|---|--:|
| as shipped | 2232 µs |
| `query.data` cleared before push | **356 µs** |

an **84% reduction**, consistent with the 1224 µs measured free cost. The rest of the
breakdown is unchanged (entry 75 -> 72 µs, handoff 146 -> 134, training 701 -> 697 ms).

> **A measurement caution, recorded because it nearly buried this.** An earlier version of
> this section asserted the deallocation mechanism from an r = +0.996 correlation alone; it
> was then *retracted* when a microbenchmark showed freeing 113 MB costing 0.6 µs. That
> benchmark was wrong: it freed `torch.empty` buffers whose pages were never touched, so
> there was nothing resident to return. Filling the tensor first — as the dataloader does —
> moves the cost from 0.7 µs to 1224 µs. A microbenchmark that does not reproduce the
> workload's page-residency does not measure the workload's free cost.

**Not applied to the shipped framework.** The one-line change would cut ~1.9 ms/query at
S b64 on the M2 Pro and reduce peak footprint, but it changes what E2 measures and would mean
re-collecting a third time. The measurement above is a standalone probe; the shipped code is
unchanged and the reported numbers are of the framework as it stands.

So: "the framework's cost is a fixed O(1) tax" is supported on GB10 and **not** on the M2
Pro, where it carries a 16 µs/MB payload term the framework could avoid. The amortization
claim holds on both.

### A polling artifact, found and removed

The E2 breakdown is what caught it. `pipeline.py` spanned every poll of its result queue and
waited in 100 ms slices on the drain path, so with `serialize_queries` both threads woke
`2 x L_q / 100 ms` times per query — 11 times per query at b64 — each contending for the GIL
with the stage threads whose duration is the measured quantity. The monolith has neither
thread.

Measured on GB10 b64, changing only that:

| | `dl - gap` | cost of decomposition |
|---|--:|--:|
| 100 ms polling | +3702 µs | +5912 µs |
| condition variables | **+353 µs** | **+4830 µs** |

E1 is unaffected — its queries complete in µs-ms, so its collector never sat through a
timeout. Confirmed by an interleaved same-session A/B on pinned GB10: paired deltas of
-1.55 µs at depth 1 and -22.98 µs at depth 10, both straddling zero.

### Gates

| gate | outcome |
|---|---|
| 11 repetitions per cell per configuration, zero failures, no truncated CSVs | **pass**, verified from disk |
| span count constant per run, independent of query latency | **pass** — exactly 2401 on all 99 traced runs on *both* machines (it was latency-proportional and machine-dependent before the fix) |
| no negative intervals in the breakdown | **pass**, no run excluded |
| Δ identity closes within a stated tolerance | **pass with tolerance stated**: +74 to +1321 µs, sources above |
| no negative cost of decomposition without explanation | **pass on GB10** (none negative); **explained on M2 Pro** (3 cells, below resolution) |

One collection artifact was caught by the row-count check and defused: a run killed
part-way left a partial CSV that the next run with the same label appended to
(`mod_meffv2s_b64_choreo_gb10_r1`, 955 rows, carrying a 4.6-hour interval). `analyze_e2.py`
takes only the last session, and `collect_e2.sh` now clears a stale file on both sides.

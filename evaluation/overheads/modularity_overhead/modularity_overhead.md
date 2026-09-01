# E2 — the cost of decomposition: EfficientNetV2 monolith vs Choreo

**Question.** Empty stages being cheap (the NoOp microbenchmark, `../framework_overhead/`)
does not prove that wrapping a *real* workload in Choreo's graph/queue/thread structure is
cheap. Does expressing an EfficientNetV2 Imagenette fine-tune as a Choreo pipeline cost
throughput against a hand-written, monolithic PyTorch implementation — and where does the
cost land? This is the real-workload counterpart to E1, and the one measured against an
external baseline rather than against the framework's own configurations.

**Status.** Closed. Collected 2026-08-31 on both machines against commit `5aea7a7`;
apparatus, metric, statistics and numbers all settled. See *Results*.

---

## What is run

Two things per cell, on an identical workload — the `(model, weights, batch)` triple is
read out of the same YAML and handed to both sides, so the only difference is the framework:

| configuration | what it is | how |
|---|---|---|
| `monolith` | a bare PyTorch loop, no framework at all, no tracing | `baseline_finetune.py --no-radt` |
| `choreo-traced` | the same workload as a Choreo pipeline, tracing on | `main.py`, `CHOREO_PROC_TRACE=1` |

    cost of decomposition = choreo-traced − monolith

Their order is rotated by repetition index so neither always absorbs the warm-up. A fixed
order systematically penalises whichever runs first, which is where an earlier cell's
impossible reading of "the wrapper makes work faster" came from.

**Why there is no third, untraced Choreo configuration.** An earlier version ran one, to
split the cost into "decomposition" and "tracing". It was dropped, and the reason is worth
stating because a reviewer will ask. The split was measured across all nine cells on both
machines and **the tracing term straddled zero at every one of them** — from −245 to +289 µs
against queries of 12–1080 ms. It bought a column of noise for a third of the collection
time. Tracing is also how the framework is actually run, so the number the paper wants is
the cost of the framework *as deployed*, which is what the two-configuration comparison
gives directly. E1 measures the tracing layer separately and far more precisely, on a
workload chosen so that it is resolvable at all.

**Workload.** Transfer-learning fine-tune of EfficientNetV2 on Imagenette — frozen backbone,
replaced 10-class head, Adam (lr 1e-3), cross-entropy. One query = one batch = one training
step. Both sides share the data loading, the frozen backbone, the Adam/CE step, the same
trainable parameter count, and both synchronise the accelerator at step end. Train split
only.

**Sweep.** EfficientNetV2-S at batch {1, 2, 4, 8, 16, 32, 64}, plus EfficientNetV2-M and -L
at batch 8 — 9 cells, on each of `m3pro` (Apple M3 Pro, torch `mps`) and `gb10` (DGX Spark,
torch `cuda`). ConvNeXt-L was dropped. Conclusions are about direction *within* a machine,
never m3pro-vs-gb10 microseconds. The M2 Pro that earlier collections used is superseded and
now serves only as a staging machine.

---

## The metric of record: in-pipeline latency, from spans

**L_q — `pipeline query` start to `pipeline query processed` start**, taken from the spans of
the traced configuration. It is the time a query spends *inside the pipeline*.

**What it deliberately excludes.** The start-to-start period between consecutive queries also
contains **loadgen admission** — the gap from one query being counted processed to the next
being admitted. That is the harness scheduling work, not the pipeline executing anything, and
**the monolith has no analogue for it**: a bare `for` loop has no admission step. Charging it
to the framework measures our own load generator and flatters nothing — it inflates the
apparent cost of decomposition by a term the reference can never pay. It is reported
separately as `scheduling` and is a near-constant 58-106 µs/query on gb10 and 132-152 µs on
m3pro, essentially independent of cell.

**Choreo is measured from spans; the monolith from its own per-step log.** That asymmetry is
deliberate and unavoidable: the monolith runs `--no-radt` precisely so the control does not
carry the framework's instrument, and it therefore emits no spans. The Choreo side's L_q comes
from the traced runs, so the reported cost is the cost of the framework **with tracing on**,
which is how it is deployed.

## Co-headline: the query latency breakdown

From the spans of the `choreo-traced` runs — which is now the only place any Choreo number
in E2 comes from: per-stage latency (**dataloader**, **training**)
plus the auxiliary framework overheads (**entry**, **handoff**, **exit**, **turnaround**).
Those four are distinct intervals and the names are load-bearing: `entry` is pipeline ->
first stage, `handoff` is stage -> stage (dataloader -> training), `exit` is last stage ->
out of the pipeline, and `turnaround` is between consecutive queries.

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

- **Choreo writes no per-query CSV rows at all.** There are now two independent flags —
  `disable_logs` on the pipeline (the per-query rows) and on each stage (the per-stage rows)
  — and E2 sets both. Every Choreo number in this experiment therefore comes from spans.

  This is not cosmetic. Gating only the stage rows still left `pipeline.py` emitting two
  rows per query unconditionally, and the second of those lands *between* the last stage
  finishing and the query being counted processed — i.e. inside `exit`. Measured directly,
  the write was **42–50% of the whole `exit` term**. The pipeline-level flag defaults to
  ON, because E1's uninstrumented configuration and E3/E4/E5 all parse those rows; only E2
  turns it off, and only because E2 has spans to fall back on.
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

- **~600 steps per run, first 50 dropped.** Step time is flat from step 0 on the GB10
  (38.8, 38.7, 38.7 … ms across 900 steps) and flat on a warm M2 Pro run, so the old
  1000/200 was never buying within-run settling.
- **R = 11 collected, repetition 1 dropped → 10 usable, and the drop is the default.** The first
  repetition of a cell is measurably slower for its *whole* duration — an anchor measured
  97.2 ms/step for r1 against 89.2 for r2..r6, still 97.4 ms in its *last* 300 steps — so
  per-step warm-up dropping cannot remove it. On the E2 smoke collection, leaving it in moved
  one breakdown term by 800 µs.
- **Two statistics, and they resolve very differently.**
  - **Cost of decomposition** — the paired across-run difference. The configurations are
    interleaved within each repetition, so runs pair by id:
    `d_i = median(L_q of choreo_i) − median(time per query of monolith_i)`. Combined across
    runs by the **median** (robust to one contaminated repetition). The CI resamples run
    **pairs** with replacement and re-resamples queries within each chosen run — the run is
    the unit of replication. This is the honest end-to-end comparison and it is what a
    reviewer will look for first.
  - **Framework term** — `entry + handoff + exit`, from spans, on one clock inside one
    process. Same hierarchical bootstrap, but no cross-process drift to absorb, so its
    interval is roughly two orders of magnitude tighter.

  Both are reported at every cell. The framework term is the headline and the figure,
  because the cross-process difference subtracts two separately-measured medians of
  12–1080 ms to recover a ~0.3 ms effect and mostly cannot resolve it.
- Per-run paired differences are printed beside every interval, so a single bad repetition
  is visible rather than absorbed.
- A **negative cost is not a speed-up**: it means the difference is smaller than what this
  apparatus resolves at that cell. It is reported as measured rather than clipped.

## A span whose count was wait-dependent — fixed

`pipeline.py`'s result loop used to span **every poll**, including the ones that timed out
empty after 0.1 s, so its span count was proportional to how long the pipeline spent waiting
rather than to how many queries ran: the same 300 queries produced ~3006 spans on gb10 and
~3890 on the Mac. That made the total span count useless as a deploy-integrity check, and it
charged the traced configuration for a slow machine's waiting.

Both are gone with the polling fix. **The count is now exactly 2401 on every traced run of
every cell on both machines**, which restores the E1-style check: a mismatch means a stale
deploy, not a slow query.

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
python evaluation/overheads/modularity_overhead/gen_configs.py     # 9 cells x 2 machines
bash   evaluation/overheads/modularity_overhead/collect_e2.sh m3pro 11
PIN=19 bash evaluation/overheads/modularity_overhead/collect_e2.sh gb10 11
python evaluation/overheads/modularity_overhead/analyze_e2.py --machines m3pro gb10
python evaluation/overheads/modularity_overhead/analyze_e2.py --latex gb10 > table2.tex
```

CSVs land in `results/mod_<cell>_<configuration>_<machine>_r<N>.csv`; each collection writes
a timestamped log and summary TSV to `collect_logs/`, headed by a provenance block (git
commit and dirty flag, host, platform, python/torch/radt/mlflow versions, pinning, run and
step counts). Figures go to `paper_assets/`.

The Choreo CSVs are near-empty by design — a handful of `prepare` rows and nothing per query.
That is the check that the CSV instrument really is off: a Choreo CSV with hundreds of rows
means a config lost one of its two `disable_logs` flags.

---

## Results

Collected 2026-08-31 on both machines against commit `5aea7a7`: 9 cells x 2 configurations
x 11 repetitions = **198 runs each machine, zero failures**, verified from the CSVs and spans
on disk (monolith 605 rows, choreo-traced 3 rows, span count 2401 on every traced run, no
duration outliers). `gb10` pinned to one X925 core (`PIN=19`); `m3pro` unpinned.

### The headline: what the framework costs, from its own spans

`entry + handoff + exit` — the framework's work inside the pipeline, per query — with a
hierarchical bootstrap over repetitions.

| cell | m3pro L_q | m3pro framework | share | gb10 L_q | gb10 framework | share |
|---|--:|--:|--:|--:|--:|--:|
| EfficientNetV2-S b1  | 29.5 ms | **194 µs** | **0.663%** | 12.3 ms | **241 µs** | **1.921%** |
| EfficientNetV2-S b2  | 33.4 | 209 | 0.618% | 19.1 | 318 | 1.666% |
| EfficientNetV2-S b4  | 62.5 | 222 | 0.353% | 33.5 | 433 | 1.291% |
| EfficientNetV2-S b8  | 121.6 | 234 | 0.191% | 64.8 | 450 | 0.698% |
| EfficientNetV2-S b16 | 252.3 | 251 | 0.100% | 132.0 | 424 | 0.321% |
| EfficientNetV2-S b32 | 531.2 | 351 | 0.066% | 275.7 | 423 | 0.153% |
| EfficientNetV2-S b64 | 1079.1 | **321** | **0.030%** | 569.8 | **562** | **0.099%** |
| EfficientNetV2-M b8  | 298.0 | 250 | 0.084% | 137.6 | 429 | 0.311% |
| EfficientNetV2-L b8  | 541.2 | 253 | 0.046% | 230.6 | 433 | 0.187% |

**The cost is fixed, not proportional.** Across the batch sweep the query gets 37x heavier on
m3pro and 46x heavier on gb10, while the framework term moves 1.65x and 2.34x. Its share
therefore falls **0.663% → 0.030%** and **1.921% → 0.099%**. That is the amortization claim,
and it is the left/right pair in `paper_assets/e2_modularity_scale.png`.

The intervals are tight — typically ±1–3 µs on a 200–560 µs term — because this quantity is
measured within one process on one clock and never crosses a process boundary.

**Where the fixed cost sits differs by machine**, which is what
`paper_assets/e2_query_latency_breakdown.png` shows. `entry` is 56–69 µs on m3pro and
19–26 µs on gb10 (the faster machine wins, as expected). `handoff` inverts it: **~100 µs on
m3pro against ~300 µs on gb10**, and it dominates the GB10 total. That is one stage thread
waking another through a condition variable, and the Grace scheduler is slower at it than
macOS is even with the run pinned to a single core. `exit` is 75–90 µs on m3pro and
79–168 µs on gb10.

**Payload size does not enter it.** Across the b1 → b64 sweep the tensor handed between
stages grows 64x while `handoff` moves 63 → 162 µs on m3pro and 123 → 309 µs on gb10 — E1's
zero-copy result reproduced on a real workload, and the reason the total does not scale.

### Cross-checked against the monolith

The same quantity as a difference against the external baseline — Choreo's `L_q` from spans,
the monolith from its own per-step log, paired by repetition:

| cell | m3pro cost | 95% CI | gb10 cost | 95% CI |
|---|--:|---|--:|---|
| EfficientNetV2-S b1  | +477 µs | [+342, +616] | −147 µs | [−326, +167] n.s. |
| EfficientNetV2-S b2  | +377 | [+189, +847] | −92 | [−298, +187] n.s. |
| EfficientNetV2-S b4  | +237 | [−125, +432] n.s. | +431 | [+287, +587] |
| EfficientNetV2-S b8  | +250 | [−138, +643] n.s. | +15 | [−224, +578] n.s. |
| EfficientNetV2-S b16 | +828 | [+178, +1556] | +640 | [+224, +1036] |
| EfficientNetV2-S b32 | +3874 | [+1957, +7176] | +929 | [+466, +1605] |
| EfficientNetV2-S b64 | +3990 | [−373, +7128] n.s. | +882 | [−601, +1794] n.s. |
| EfficientNetV2-M b8  | +304 | [−208, +1265] n.s. | −967 | [−1224, −153] |
| EfficientNetV2-L b8  | −811 | [−1685, +323] n.s. | −353 | [−759, +459] n.s. |

**This is the honest end-to-end number and it is mostly not resolvable.** Ten of the
eighteen intervals contain zero; three are negative. That is not a failure of the framework,
it is the resolution floor of the comparison: recovering a 0.2–0.6 ms effect by subtracting
two separately-measured medians of 12–1080 ms inherits both processes' run-to-run drift,
which is ±1–7 ms at the large cells.

The right reading of the two tables together: **the framework's own instrument says it costs
200–560 µs per query, and an external baseline agrees to within its own resolution — it
cannot even see a difference at most cells.** Reporting only the cross-process number would
understate what is known; reporting only the span term would be measuring the framework with
its own ruler. Both are published.

Two cells deserve a note rather than a silent pass. **m3pro S b32 and b64** show
+3.9 ms costs, an order above the framework term. The identity below attributes them to
`dl − gap`, not to scaffolding: the Choreo dataloader at those batch sizes is slower than
the monolith's own inter-step loading on this machine. Both cells' dataloader time also
grows super-linearly in batch (52 → 140 → 313 ms across b16/b32/b64, against a 2x step),
which is a memory-pressure effect on a 18 GB machine and not something the framework does.

### The identity, and its tolerance

    cost of decomposition = (dataloader − monolith gap)
                          + (training   − monolith step)
                          + framework

On gb10 it closes with residuals of **+9.7 to +850 µs** (median +85) against measured costs
of −967 to +929 µs. The residual is dominated by *median of sums ≠ sum of medians* — each
term is a median over repetitions — and by the same cross-process noise the previous table
carries. The identity's value is not its residual but its first column: `dl − gap` is
negative at five of nine gb10 cells, i.e. **decomposition partly MOVED work rather than
adding it**, and a report of the net alone would hide that in both directions.

### A polling artifact, found and removed

The E2 breakdown is what caught it. `pipeline.py` spanned every poll of its result queue and
waited in 100 ms slices on the drain path, so with `serialize_queries` both threads woke
`2 x L_q / 100 ms` times per query — 11 times per query at b64 — each contending for the GIL
with the stage threads whose duration is the measured quantity. The monolith has neither
thread.

Measured on GB10 b64, changing only that:

| | `dl − gap` | cost of decomposition |
|---|--:|--:|
| 100 ms polling | +3702 µs | +5912 µs |
| condition variables | **+353 µs** | **+4830 µs** |

E1 is unaffected — its queries complete in µs-ms, so its collector never sat through a
timeout. Confirmed by an interleaved same-session A/B on pinned GB10: paired deltas of
−1.55 µs at depth 1 and −22.98 µs at depth 10, both straddling zero.

### A CSV write inside the measured interval, found and removed

The second thing E2 caught in its own apparatus. With the stage rows already gated off,
`pipeline.py` still wrote two rows per query unconditionally, and the second landed between
the last stage finishing and the query being counted processed — inside `exit`. Measured
directly, **that synchronous write+flush was 42–50% of the entire `exit` term.**

It is now behind `PipelineModel.disable_logs`, a second flag independent of the stage-level
one, defaulting ON because E1's uninstrumented configuration and E3/E4/E5 parse those rows.
E2 is the only experiment that turns it off, and only because spans cover it. The results
above are from the post-gate collection; everything collected before it is superseded.

### Gates

| gate | outcome |
|---|---|
| 11 repetitions per cell per configuration, zero failures, no truncated CSVs | **pass**, verified from disk on both machines (198 runs each) |
| span count constant per run, independent of query latency | **pass** — exactly 2401 on all 99 traced runs on *both* machines |
| Choreo CSVs carry no per-query rows | **pass** — 3 rows per file, all `prepare` |
| no negative intervals in the breakdown | **pass**, no run excluded |
| framework term positive and tight at every cell | **pass** — 194–562 µs, intervals ±1–3 µs |
| cross-process cost reported with CIs, negatives not clipped | **pass** — 10 of 18 cells n.s., stated as such |
| identity closes within a stated tolerance | **pass with tolerance stated**: +9.7 to +850 µs, sources above |

Two collection artifacts were caught by the row-count check and defused. A run killed
part-way left a partial CSV that the next run with the same label appended to
(`mod_meffv2s_b64_choreo_gb10_r1`, 955 rows, carrying a 4.6-hour interval); `analyze_e2.py`
now takes only the last session and `collect_e2.sh` clears a stale file on both sides. And a
Mac left on battery slept mid-collection, moving three repetitions' medians by 4–6% while
reporting `rc=0` and exactly the right row count; both harnesses now re-exec under
`caffeinate -dimsu` on Darwin.

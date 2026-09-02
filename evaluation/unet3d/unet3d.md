# E3 — MLPerf 3D-UNet / KiTS19: reproduce the standard, then show what it hides

**Question.** Two prongs, and the order is the point.

1. **Parity, on GB10 only.** Run MLPerf's *own reference harness* and Choreo's port of the
   same 3D-UNet/KiTS19 workload on the same machine, and show Choreo matches it on
   **accuracy (DICE) and performance**. This is a same-device faithfulness check. It earns
   the right to criticise the standard; without it, prong 2 reads as a strawman.
2. **The measurement boundary, on both machines.** MLPerf preprocesses the dataset
   **offline** — its QSL preloads `.pkl` volumes — and times only inference. That is valid
   for offline batch. Online, a request arrives with its own raw volume: there is nothing to
   prefetch, so load and preprocess sit unavoidably on the per-request critical path. Choreo
   times the whole graph and reports the share MLPerf reports as zero.

**Status.** Closed. Reworked to the E1/E2 shape 2026-09-01, collected 2026-09-02 on both
machines. Prong 1 parity holds on accuracy (0.86329 vs 0.86168) and performance (−0.1% on
median inference, against a `VALID` reference run); prong 2 measures a hidden share of 6.3%
on m3pro and 15.8% on gb10. See *Results*.

---

## What is run

Two configs per machine, and they are never mixed.

| config | stages | writes per query | timed? |
|---|---|---|---|
| `perf` | loader → preprocess → inference | **nothing** | yes, R=6 (repetition 1 dropped) |
| `acc` | the same three plus `KiTS19DiceScore` | one CSV row per case | **never**, R=1 |

**Why two.** Output handling does not belong in a timed pipeline. The previous end stage,
`TerminalCapture`, `repr()`s whatever sits in `query.data` — here a multi-hundred-megabyte
numpy segmentation — so keeping it would put output serialisation inside the measurement,
which is the same class of error as the CSV write E2 just removed. MLPerf itself separates
`AccuracyOnly` from `PerformanceOnly`, so E3 mirrors it.

A useful side effect of dropping the capture stage: **inference becomes the pipeline's exit
stage**, so the framework's payload-release rule frees the segmentation on the inference
thread instead of carrying it further.

**Why the scorer is a stage and not a script.** `TerminalCapture` cannot produce a DICE
number — its JSONL carries a truncated `repr()` of the array, not the array. The segmentation
only exists inside the pipeline, so scoring has to happen there. `KiTS19DiceScore` is a port
of the reference harness's `get_dice_score`, smoothing terms included, scoring the prediction
against the label that went through the *same* resample/crop/pad chain. Without it the parity
gate is a number in a log file that nobody can re-derive.

**Timing: spans only.** The `perf` configs set `disable_logs` on the pipeline *and* on all
three stages, so nothing is written per query and every number comes from spans. A `perf`
CSV that holds more than its two `prepare` rows means a config lost a flag — that is the
check, and `collect_e3.sh` asserts the flags before it runs anything.

**Workload.** MLPerf's 3D-UNet (nnU-Net) on KiTS19, SingleStream, 42 cases from
`inference_cases.json`, `serialize_queries: true`, `queue_depth: 1` — one request in flight,
end to end, which is the online-serving regime the claim is about. The numeric path is a
faithful numpy/scipy port of the reference and is deliberately left alone: parity depends on
it.

**Data and model are external dependencies**, machine-local and gitignored: KiTS19 raw
(3.8 GB, 43 cases) under `data/kits19/raw`, and the TorchScript checkpoint (119 MB, Zenodo
5597155) under `models/3dunet_kits19/`. `stages/unet3d_kits19/BUILD.md` has the acquisition
steps. The MLPerf reference harness itself is a clone of `mlcommons/inference` under
`scratchpad/`, also machine-local.

---

## E3 runs UNPINNED, unlike E1 and E2

E2 ended up single-core-pinned on GB10 because it was both cleaner and faster there. **E3
must not inherit that, and the reason is bias rather than hygiene.**

E3's claim is the *ratio* of CPU preprocessing to GPU inference. Pinning to one core throttles
`scipy.ndimage.zoom` — the dominant preprocessing cost — while leaving GPU inference
untouched. That inflates the preprocessing share, which is the number we are arguing is
larger than MLPerf admits. **Pinning would manufacture our own result.**

Choreo and the MLPerf reference must therefore see identical CPU conditions, and that
condition is unpinned. `collect_e3.sh` refuses a `PIN` rather than honouring it. A reader
comparing E2 and E3 will notice the inconsistency; it is explained here so they find the
answer before they have to ask.

---

## The metric, and the decomposition

Per query, from the generic framework spans of a three-stage graph:

    pipeline query -> loader.run -> loader.push -> preprocess.run -> preprocess.push
                   -> inference.run -> inference.push -> pipeline query processed

giving `entry`, `load`, `handoff_lp`, `preprocess`, `handoff_pi`, `inference`, `exit`. They
are successive instants within one query on one clock, so they are non-negative by
construction and sum to L_q exactly; a negative one means the spans are mis-paired and the
run is refused rather than medianed over.

**`load` is near-zero by design.** `KiTS19CaseLoader` emits only the case id and its file
paths. The read, decompress, resample, normalize and pad all happen inside `preprocess` —
which is exactly where MLPerf's offline QSL does them. The share the boundary hides is
therefore essentially the preprocess stage.

**`entry` + the two hand-offs + `exit` is the framework's own scaffolding.** E2 measures it
directly at 0.2–0.6 ms per query. Against a ~6 s query here it is four orders of magnitude
down, and it is reported only so the components provably sum to L_q.

**The independent variable is recorded as a span attribute.** `KiTS19Preprocess` emits one
`case_size` marker per query carrying `n_subvolumes` and the post-resample `image_shape`.
KiTS19 volumes tile into 8–144 sliding-window sub-volumes, an ~18x range, and that is what
makes the hidden share vary case to case. The previous version of E3 recovered these by
joining against a *different experiment's* CSV; now each case's size is a property of its own
trace.

**Store: res17, not a local sqlite file.** The local-store exemption covers the overhead
experiments (E1, E2) only — they emit spans at a rate no real workload approaches. E3 is a
real workload at seconds per query and a handful of spans per query, which is what the remote
server is there to carry.

---

## Statistics

- **R = 6 collected, repetition 1 dropped → 5 usable.** The first repetition of a cell is
  measurably slower for its whole duration, so per-query warm-up dropping cannot remove it.
- **No per-query warm-up drop.** There are only 42 queries and the case order is fixed, so
  dropping the head would drop *specific cases* — and case identity is the independent
  variable here. The whole run is kept.
- **CIs are a hierarchical bootstrap with the run as the unit of replication**, as in E1/E2:
  repetitions resampled with replacement, then queries within each chosen repetition.
- **Per case first, aggregated after.** KiTS19 cases are not interchangeable. Every prong-2
  claim is computed per case and only then summarised, and the distribution is reported — the
  median is what gets quoted, with the maximum named as the range endpoint it is.

---

## Known objections, and where each is answered

From the mock review that moved this section from Weak Reject to Weak Accept:

- **"MLPerf isn't blind — there's a Server scenario."** True, and the honest statement is
  narrower: the *3D-UNet/KiTS19 benchmark* ships only Offline and SingleStream. Say that, not
  "MLPerf has no online scenario".
- **"Preprocessing-grows is a CPU-frozen artifact — DALI and pipelining dissolve it."** The
  claim is about the **measurement boundary**, not about an unoptimisable stage. Prefetching
  can hide preprocessing for offline batch; nothing can prefetch a request that has not
  arrived. Do not argue that the stage is hard to optimise.
- **"70% is a per-study max, not representative."** Report the distribution and the median;
  use the max only as the range endpoint.
- **Prong 1 is GB10-only** and must never be presented as a cross-device claim.
- **The device comparison is a mismatch in speedups, not a claim about equal CPUs.** GB10's
  GPU pulls much further ahead of the Mac's than its CPU does, so the un-accelerated stage
  dominates more exactly where the accelerator is better. State it that way.

---

## Reproduce

```bash
# 1. accuracy pass — R=1, never timed, writes results/dice_<machine>.csv
bash evaluation/unet3d/collect_e3.sh gb10  1 acc
bash evaluation/unet3d/collect_e3.sh m3pro 1 acc

# 2. timed passes — R=6, repetition 1 dropped at analysis
bash evaluation/unet3d/collect_e3.sh gb10  6
bash evaluation/unet3d/collect_e3.sh m3pro 6

# 3. analysis — reads spans from res17, writes tables and both figures
python evaluation/unet3d/analyze_e3.py --machines m3pro gb10
```

The harness records to res17 (experiment 138) and requires `MLFLOW_TRACKING_URI` to be set;
the credentials live as conda env config vars, so activate the environment rather than
invoking the python binary directly. Results land in `results/<machine>/`, a timestamped log
and summary TSV in `collect_logs/` headed by a provenance block, figures in `paper_assets/`.

**Analysis runs wherever the results are.** `analyze_e3.py` reads the timing from spans on
res17, so that half works from anywhere, but the DICE table is a local file the accuracy pass
wrote on the collecting machine. Pull it before analysing off-machine:

```bash
rsync -a babyxena:collocation-benchmark/evaluation/unet3d/results/dice_gb10.csv \
         evaluation/unet3d/results/
rsync -a itu-mac:collocation-benchmark/evaluation/unet3d/results/dice_m3pro.csv \
         evaluation/unet3d/results/
```

Without them the accuracy row reads `—` and the analyzer says so rather than skipping the
table.

**The valid reference run took two attempts, and the second trap is worth recording.**
`logs_perf/` is the original and says `Result is : INVALID` — `user_e3.conf` capped
`max_query_count` at 43, the size of the QSL, so loadgen never reached early stopping.
Lifting the cap was not enough: `min_query_count` is a **floor, not a cap**, and loadgen
computes `effective_min_query_count` from `min_duration / expected-per-query-latency`, taking
whichever is larger. With no `target_latency` set it assumed a tiny query and demanded
**120013** of them to fill 600 s; the run was still going 8.5 hours and 4087 queries later,
with a 1.8 GB trace. Setting `*.SingleStream.target_latency = 8000` (3D-UNet runs ~7.5 s/case
on gb10) brings the effective minimum to 172 and the run to ~20 minutes. The working config
is kept at `mlperf_reference/user_valid.conf`.

---

## Results

Collected 2026-09-02 against commit `bccefe8`. **R = 6 timed runs per machine, repetition 1
dropped, 5 usable**, plus one accuracy pass each. Zero failures; every run emitted exactly
463 spans and 3 CSV rows, and the run times were stable to 0.5% (m3pro 2577-2591 s) and 0.7%
(gb10 414-417 s).

`analyze_e3.py` regenerates every number and both figures from spans; the tables below are
its output, kept here so the write-up and the analyzer cannot disagree.

### Prong 1 — parity with the MLPerf reference, on GB10

**Accuracy.** The reference run is `mlperf_reference/logs_perf_valid/`, and it reports
`Result is : VALID` — 172 queries, min duration, min queries and early stopping all
satisfied.

| harness | cases | mean DICE | kidney | tumor |
|---|--:|--:|--:|--:|
| MLPerf reference | 43 | 0.86168 | 0.9347 | 0.7887 |
| Choreo, gb10 | 42 | **0.86329** | 0.93418 | 0.79241 |
| Choreo, m3pro | 42 | **0.86330** | 0.93418 | 0.79242 |

MLPerf's gate is 99% of 0.86170 = **0.85308**; Choreo clears it. The two devices agree to
five decimal places and differ by at most 7e-5 per case, so the cross-device comparison in
prong 2 is not confounded by numerics.

**Performance**, inference only — the part MLPerf times:

| harness | median (ms) | mean (ms) | p90 (ms) |
|---|--:|--:|--:|
| MLPerf reference | 5904 | 7669 | 14581 |
| Choreo (inference stage) | 5899 | 7923 | 14558 |

**Median inference latency differs by −0.1%**, p90 by −0.2%. Matched per case over all 42
cases both harnesses ran: median **−0.2%**, mean +7.0%, range −0.4% to +82.0% — and after
correcting the four warm-up cases below, median **−0.2%**, mean **+0.5%**, range −0.8% to
**+11.3%**.

### A device warm-up transient, and what it does not affect

The mean and the maximum above are carried by the **first four queries of the run and only
those** — case_00000 +82%, case_00003 +75%, case_00005 +61%, case_00006 +66%. Over the
remaining 38 cases: median **−0.2%**, mean **+0.2%**, range −0.4% to +8.5%.

It is a device property, not a framework cost, and three things establish that:

- it is **present on gb10 and absent on m3pro** — every position there is within 0.4% of
  steady state — while the pipeline is byte-identical on both;
- it tracks **position in the run, not input shape**: a repeated shape at position 1 is still
  60-80% slow, and a brand-new shape at position 4 is already at steady state. That rules out
  per-shape kernel autotuning, which was the obvious first hypothesis and is wrong;
- it reproduces in all five repetitions (case_00000: 10756, 10739, 10473, 10455, 10811 ms).

The mechanism is **not otherwise characterised here**, and it should not be guessed at in the
paper. What matters is bounded: it affects 4 of 42 cases, and the median statistics of both
prongs are unmoved by it.

The reference does not show it because loadgen issues more queries than the QSL holds, so
each case is sampled several times (4x here) and its own median discards the cold occurrence.
That is a **harness asymmetry**.

**Corrected, without re-running the whole collection.** Only the first four queries are
affected and everything after is flat, so the correction only needs those four. The config
`warmup_cases.json` runs the four affected cases **twice**: pass 1 reproduces the original
head-of-run conditions exactly (same cases, same order, so the warm-up is identical by
construction rather than approximated), pass 2 measures them warm, and two already-clean
cases follow as controls. R=3, ten queries per run, ~2.5 minutes each against ~7 for a full
42-case run.

It self-validates. Pass 1 reproduced the transient — +79.7%, +69.5%, +61.5%, +61.9% against
the originally measured +82%, +75%, +61%, +66% — and the controls came back at −0.1% and
−1.1% of steady state, so the prefix genuinely warmed the device.

| case | n | as collected | warm-corrected |
|---|--:|--:|--:|
| case_00000 | 50 | +82.0% | **−0.8%** |
| case_00003 | 50 | +74.6% | **−0.6%** |
| case_00006 | 64 | +65.7% | **−0.3%** |
| case_00005 | 108 | +61.2% | **+11.3%** |

Three of the four collapse to within 1% of the reference, which is what the warm-up
explanation predicts.

**`case_00005` does not, and that residual is an open item.** It is +11.3% above the
reference when measured warm, so it is not warm-up. It is also not volume size: across the
38 steady-state cases, ms per sub-volume *falls* with volume (rho = **−0.75**, fixed per-case
overhead amortising over more sub-volumes), and the largest volumes in the set — 320x448x448,
64.2 Mvoxel, 144 sub-volumes — run at 116 ms/sub-volume, at steady state, while case_00005 at
51.4 Mvoxel runs at 130. Two hypotheses tested and refuted; no third is asserted here. It is
one case in 42, the median parity statistic is unaffected, and it is recorded rather than
explained away.

### Prong 2 — what the measurement boundary hides

| machine | R | end-to-end L_q | load | preprocess | inference | framework | **hidden share** |
|---|--:|--:|--:|--:|--:|--:|--:|
| M3 Pro (mps) | 5 | 50889 ms | 0 | 3990 | 45215 | 0.4 | **6.3%** [6.2, 6.5] |
| GB10 (cuda) | 5 | 8950 ms | 0 | 1475 | 5899 | 1.5 | **15.8%** [14.8, 16.4] |

The framework's own scaffolding is **0.4-1.5 µs** against requests of 9-51 seconds — seven
orders of magnitude down, and listed only so the components provably sum to L_q.

**The share is not one number.** Per case:

| machine | median | p25 | p75 | min | max |
|---|--:|--:|--:|--:|--:|
| M3 Pro (mps) | 6.3% | 5.4% | 8.2% | 1.7% | 27.1% |
| GB10 (cuda) | 15.7% | 13.0% | 20.8% | 8.5% | 68.4% |

Quote the median. The maximum is the endpoint of a range, and quoting it alone is exactly the
objection this table exists to answer.

**Why it is larger on the faster device.** Inference is **7.7x** faster on gb10; preprocessing
only **2.7x**. The un-accelerated stage dominates more precisely where the accelerator is
better. This is a mismatch in speedups, **not** a claim that the two machines have equal CPUs.

| machine | preprocess | inference | sub-volumes (median) | ms per sub-volume |
|---|--:|--:|--:|--:|
| M3 Pro (mps) | 3986 ms | 45206 ms | 50 | 905 |
| GB10 (cuda) | 1470 ms | 5894 ms | 50 | 118 |

### The share against case size, and a model that had to be replaced

| machine | n range | share at min n | share at max n | Spearman rho | preprocess | inference | asymptote |
|---|--:|--:|--:|--:|---|---|--:|
| M3 Pro (mps) | 8-144 | 27.1% | 4.5% | **−0.38** | 41.9n + 1344 | 914n − 390 | **4.4%** |
| GB10 (cuda) | 8-144 | 68.4% | 9.9% | **−0.56** | 10.3n + 1031 | 122n + 256 | **7.8%** |

**More sub-volumes means a SMALLER share**, not a larger one. The independent pre-rework
dataset agrees (rho = −0.50, 66.3% at n=8 falling to 11.5% at n=144, median 16.6% against
15.7% here). What *is* positive is absolute preprocessing **seconds** (rho ≈ +0.6): bigger
cases do preprocess for longer, just not as a larger fraction.

The obvious model, `share = P / (P + S·n)` with P a fixed per-volume cost, is **refuted by
this data** and was reported before it was checked. Preprocessing is not constant — it
correlates with n at +0.77 (m3pro) and +0.60 (gb10) over a 15x and 11x range, because a
volume with more sub-volumes is a physically bigger volume to read, resample and pad. Forcing
P constant made the fit absorb that growth into S, which came out **4.8x below** the directly
measured cost per sub-volume. That disagreement is what exposed it.

Both terms are affine in n, so the share does not decay to zero. It falls towards
**P₁/(P₁+S₁)** — the ratio of the two slopes — which is **4.4%** on m3pro and **7.8%** on
gb10, and the largest cases measure 4.5% and 9.9%. That is a stronger claim than the wrong
model made: **the cost MLPerf's boundary excludes does not amortise away on large inputs; it
converges to a fixed fraction of every request.**

`rho` is −0.38/−0.56, not −0.95: case size explains the trend, not the scatter. At a fixed n
the share still varies (17.6-30.0% across four n=16 cases in the older data), because raw
file size and slice count differ independently of the post-resample tile count. The
defensible statement is "the share varies strongly across cases and part of that variation
tracks case size", not "the share is a function of n".

### Figures

- `paper_assets/e3_request_breakdown.png` — per-case latency, `preprocess` + `inference`
  stacked, one panel per machine, cases ordered by sub-volume count. The four warm-up cases
  are visible on the gb10 panel as taller inference bars among their size peers.
- `paper_assets/e3_preprocessing_share.png` — the hidden share per case, machines grouped
  side by side. gb10 is above m3pro at **every one of the 42 cases**, which is the
  cross-device claim shown per case rather than as a difference of medians.

## What was removed in the rework

Reachable through git, per the E1/E2 rule that superseded files do not stay in the tree.

- Scripts: `analyze_preprocessing.py`, `schedule_analysis.py`, `run_full_experiment.py`,
  `collect.sh`.
- Reports: `FINDING_A.md`, `REPORT.md`, `RUN_FINDING_A_ON_MAC.md`, `server_measured.md`,
  `E3_ANALYSIS.md`, `mac_server_rho099_summary.txt` — all predate the current framing, two
  cite a `run_pipelined.py` that no longer exists, and one points at figure paths retired in
  `ca72e40`.
- Data and figures at the top level: `collect_summary_{cuda,mps}.tsv`,
  `results_{cuda,mps}_r1.csv`, `preprocessing_fraction{,_stages}.{png,pdf}`.
- The scheduling work: `sched/{fifo,sjf}_cases.json` and
  `pipeline_configs/unet3d_batch_{fifo,sjf}_{cuda,mlx}.yml` — no driver, no results; the
  FIFO/SJF numbers in `REPORT.md` came from a patched MLPerf harness, not these.
- `pipeline_configs/unet3d_kits19_{cuda,mlx}.yml` — 8-case smokes without
  `serialize_queries`. (`mlx` was a misnomer throughout: the Apple path is plain torch/MPS.)
- `mlperf_gb10/` → `mlperf_reference/`; `configs/unet3d_42_{cuda,mps}.yml` → the four
  `{perf,acc}_{m3pro,gb10}` configs.

Four defects found in the survey and fixed by the rewrite: the analyzer preferred a
3-sample log over the 43-sample one (which is why matched parity showed only 16 cases); it
looked for a `logs_perf_full/` that does not exist and always fell back to the INVALID run;
`collect.sh` moved `<label>.jsonl` while `TerminalCapture` writes `<label>_outputs.jsonl`, so
the JSONL was never collected; and machine tokens were `cuda`/`mps` where the rest of the
repo uses `gb10`/`m3pro`.

One defect is deliberately **not** fixed: `inference.py` allocates two float64
`[1,3,D,H,W]` host arrays (~1.2 GB for the largest case) inside the measured inference span
and never explicitly syncs the device. That is faithful to the reference, so it is measured
rather than "fixed" — but it is there, and it should not be attributed to the framework.

---

## Collection log

**2026-09-01, m3pro (itu-mac).** Dataset staged from the M2 Pro (3.8 GB raw + the 119 MB
TorchScript model — the machine had neither). Accuracy pass then R=6 timed, unpinned, under
`caffeinate`.

**2026-09-01, gb10 (babyxena).** Accuracy pass completed clean and **closes prong 1 on
accuracy**:

| harness | cases | mean DICE | kidney | tumor |
|---|--:|--:|--:|--:|
| MLPerf reference | 43 | 0.86168 | 0.9347 | 0.7887 |
| Choreo, gb10 | 42 | **0.86329** | 0.93418 | 0.79241 |
| Choreo, m3pro | 42 | **0.86330** | 0.93418 | 0.79242 |

Kidney agrees to 0.0005 and tumor to 0.004; the mean clears MLPerf's gate (0.85308) with
room. That agreement is also what validates `KiTS19DiceScore` as the reference formula — an
earlier ad-hoc scoring of the same runs used a merged-kidney convention (kidney = classes
{1,2}) and does not reproduce the reference's per-class numbers.

**The two devices agree numerically, to five decimal places.**

| machine | cases | mean DICE | kidney | tumor | run seconds | rows | spans |
|---|--:|--:|--:|--:|--:|--:|--:|
| gb10 (cuda) | 42 | 0.86329 | 0.93418 | 0.79241 | 425 | 431 | 547 |
| m3pro (mps) | 42 | 0.86330 | 0.93418 | 0.79242 | 2602 | 431 | 547 |

Per case the two differ by at most **7e-5** (median 4e-6). CUDA and MPS produce the same
segmentation to within rounding, and the two accuracy runs emitted identical row and span
counts, so the pipelines were structurally identical as well.

That matters for prong 2, not just for tidiness: the cross-device comparison is the argument
that the hidden share grows where the accelerator is faster. If the two devices produced
materially different segmentations they would be doing different amounts of work, and any
latency difference between them would be partly that rather than the hardware. They do not.

**The gb10 TIMED pass of 2026-09-01 is contaminated and must not be used.** A MobileNetV2
training job (`train_cloud.py`) started on the same machine 26 seconds after the collection
began and held ~97 GB of GPU memory throughout.

The wall-clock column shows it without any appeal to what was running:

| run | window | seconds | spans |
|---|---|--:|--:|
| accuracy pass (before the foreign job started) | 23:32–23:39 | 425 | 547 |
| perf r1 | 23:39–00:02 | **1404** | 463 |
| perf r2 | 00:02–00:16 | **811** | 463 |
| perf r3 | 00:16–00:24 | **487** | 463 |
| perf r4 | 00:24–00:31 | **417** | 463 |

Identical work, 42 cases each, and the span count is constant at 463 throughout — nothing
about the pipeline changed. What changed is the machine: the foreign job ended at about
00:22, and the run times decay monotonically towards the idle baseline as it wound down,
**1404 → 811 → 487 → 417 s**, a factor of **3.4** across four consecutive runs of the same
work, settling at an idle baseline of ~417 s. The
accuracy pass ran before it started, does strictly *more* work per case (it resamples the
label too, and scores the result), and still finished in a third of r1's time.

A collection whose runs vary by 2.9x is not a collection with noise in it; it is a
measurement of the other job.

The span-level view says the same thing, and says *where* it lands:

| run | preprocess (median) | inference (median) | framework |
|---|--:|--:|--:|
| perf r1 | 2067 ms | 19650 ms | 1472 µs |
| perf r2 | 1776 ms | 11459 ms | 1154 µs |
| swing | **−14%** | **−42%** | — |

Inference moves three times as much as preprocessing between two identical runs. That is the
signature of GPU contention rather than general machine noise, and it fixes the direction of
the bias: with inference inflated, the share prong 2 reports is **understated**, not
flattered. The framework term is 1.2–1.5 ms — larger than E2's 0.4–0.6 ms because this graph
has three stages and two hand-offs rather than two and one, and about 0.005% of a 23-second
query either way.

E3 measures the *ratio* of CPU preprocessing to GPU inference, so a co-resident GPU job
inflates the inference stage specifically: it biases prong 2 conservatively — the hidden
share looks smaller than it is — and it breaks prong 1 outright, since the reference it is
compared against was measured on an idle machine. The foreign job was left running; it is not
ours to kill. A re-collection is queued to start once the machine goes idle, moving the
contaminated runs aside rather than deleting them.

The DICE result above is unaffected and stands: it was collected before the foreign job
started, and segmentation output is deterministic regardless of how long it takes.

This is the second time a concurrent workload has silently corrupted a collection on a shared
machine (the first was a Mac that slept mid-run). Neither was visible in the return code, the
row count or the span count. The lesson for the harness: **record machine occupancy in the
provenance header**, not just the git commit and the library versions.

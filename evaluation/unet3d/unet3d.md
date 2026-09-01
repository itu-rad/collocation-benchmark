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

**Status.** Reworked to the E1/E2 shape 2026-09-01. Collection pending.

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

**Still needed before prong 1 closes:** a **valid** MLPerf reference run on GB10. The one on
disk announces its own invalidity — `logs_perf/mlperf_log_summary.txt` says `Result is :
INVALID`, 43 queries processed against the 64 loadgen needs for early stopping. The parity
claim cannot rest on it.

---

## Results

Not collected yet. `analyze_e3.py` produces, from spans alone:

- **Prong 1**: DICE against the reference and against MLPerf's 99%-of-0.86170 gate; pooled
  inference-latency percentiles against the reference summary; and a **matched per-case**
  comparison over all 42 cases, mapping the reference's `sample_idx` to a case through the
  QSL file list the harness itself loads.
- **Prong 2**: the per-request breakdown with the hidden share, the distribution of that
  share across cases, the two stages' scaling (ms per sub-volume against per-volume
  preprocessing), and the share against case size with a `share = P / (P + S·n)` fit whose
  fitted `S` is checkable against the measured ms per sub-volume.
- **Figures**: `e3_request_breakdown.png` (per-case stacked latency, ordered by size) and
  `e3_preprocessing_share.png` (hidden share against sub-volume count, with the fit).

---

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
| Choreo | 42 | **0.86329** | 0.93418 | 0.79241 |

Kidney agrees to 0.0005 and tumor to 0.004; the mean clears MLPerf's gate (0.85308) with
room. That agreement is also what validates `KiTS19DiceScore` as the reference formula — an
earlier ad-hoc scoring of the same runs used a merged-kidney convention (kidney = classes
{1,2}) and does not reproduce the reference's per-class numbers.

**The gb10 TIMED pass of 2026-09-01 is contaminated and must not be used.** A MobileNetV2
training job (`train_cloud.py`) started on the same machine 26 seconds after the collection
began and held ~97 GB of GPU memory throughout.

The wall-clock column shows it without any appeal to what was running:

| run | seconds | spans |
|---|--:|--:|
| accuracy pass (before the foreign job started) | 425 | 547 |
| perf r1 | **1404** | 463 |
| perf r2 | **811** | 463 |

Identical work, 42 cases each, and r1 takes 73% longer than r2 — while the span count is
constant at 463, so nothing about the pipeline itself changed. The accuracy pass, which ran
before the foreign job started and does strictly *more* work per case (it resamples the label
too, and scores the result), finished in a third of r1's time.

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

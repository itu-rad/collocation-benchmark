# Running the three overhead experiments on a Mac (online)

## 1. One-time setup

```bash
conda env create -f environments/macos.yaml      # creates env `benchmark_macos` (radt async fork @9dda7b8)
conda activate benchmark_macos
```

Set the MLflow server credentials as **conda env vars** so runs log online to res17

## 2. Run the experiments (add `--online` to log to res17 with async tracing)

```bash
# Exp 1 + 2 — dispatch/queuing/threading overhead AND zero-copy context passing (one driver).
# CPU/interpreter-only, no data needed.
python evaluation/overheads/framework_overhead/run_matrix.py --runs 5 --online

# Exp 3 — real-world decomposition overhead (EfficientNetV2-S fine-tune, bare vs wrapped).
# First run auto-downloads Imagenette + weights (~1.5 GB, internet once) to tmp/.
python evaluation/overheads/modularity_overhead/run_modularity.py --device mps --runs 5 --online

# Exp 3b — MULTI-SCALE sweep: shows the fixed per-step overhead amortizes as the step grows.
# Sweeps batch (1..64 @ EfficientNetV2-S) and model (EffNetV2-S -> M -> L -> ConvNeXt-L @ batch 8).
# Run --online on an IDLE Mac: the tracing-ON arm then measures the REAL production tracing cost
# (async span export to res17; the uploader thread contends for the GIL). "Idle" isolates the
# framework overhead from other load — it does NOT mean dropping --online.
python evaluation/overheads/modularity_overhead/run_modularity.py \
    --device mps --cells evaluation/overheads/modularity_overhead/configs/scale_sweep.yml --online
# Quick smoke first (3 cells, ~10 min) to confirm it runs before the long sweep:
python evaluation/overheads/modularity_overhead/run_modularity.py \
    --device mps --runs 1 --online \
    --cells evaluation/overheads/modularity_overhead/configs/scale_sweep_min.yml
```

Drop `--online` only for offline reproduction without server access (the tracing-ON arm then
measures the cheaper local-file-store cost, not the representative production tracing cost).

## 3. What `--online` does (and a caveat)

The tracing-**ON** arm logs spans to res17 with **async export** — so the measured tracing
overhead is the *real production cost* (the background uploader thread contends for the GIL,
exactly as in the case studies), not a cheap local-file-store cost. The tracing-OFF arm is the
pure baseline. `t1 − t0` is the real tracing overhead.

**Caveat:** res17 ingests spans slower than these micro-workloads generate them, so on the
server the span *records* are partial and each tracing-ON run pays a ~30 s shutdown flush.
This does **not** affect the measurement — the timing CSV is written before the flush. The
representative tracing-overhead number comes from **Exp 3** (ms-scale, realistic span rate);
Exp 1's tracing-ON arm at µs-scale over-states tracing cost (its headline is the OFF arm).

## 4. Results + analysis

Results land in each experiment's `results/` dir (CSVs; env capture in `*_env.txt`).

```bash
# Exp 1 (dispatch, depth-flatness) + Exp 2 (zero-copy):
python evaluation/overheads/framework_overhead/analyze_noop_results.py --arm both
python evaluation/overheads/framework_overhead/analyze_payload_results.py --fig   # -> payload_zero_copy.pdf
python evaluation/overheads/framework_overhead/generate_latex_results.py > tables.tex

# Exp 3 (modularity):
python evaluation/overheads/modularity_overhead/analyze_operational_overhead.py --device mps   # headline per-step
python evaluation/overheads/modularity_overhead/true_overhead_analysis.py --device mps         # tracing layer
python evaluation/overheads/modularity_overhead/breakdown_overhead.py --device mps             # stage breakdown
python evaluation/overheads/modularity_overhead/generate_latex_results.py --device mps > table2.tex

# Exp 3b (scale sweep) — per-cell overhead table + the amortization figure:
python evaluation/overheads/modularity_overhead/analyze_scale_sweep.py --device mps --latex \
    --fig scale_sweep_amortization.pdf
```

The scale-sweep analyzer pools every cell onto one `overhead % vs step-time` curve: the absolute
µs overhead stays ~flat while its share of the step falls ~1/step_time. The worst realistic case
(EfficientNetV2-S @ batch 1) is the ceiling; every heavier cell is cheaper.

See `framework_overhead/framework_overhead.md` and `modularity_overhead/modularity_overhead.md`
for the full write-ups and reference numbers.

## 5. Rough wall-times
- Exp 1 + 2: minutes to low-tens-of-minutes (15 depths × R × 2 arms; `--online` adds ~30 s per tracing-ON run for the flush).
- Exp 3: ~30–60 min on an M2 (R=5 × baseline + 2 arms, 1100 steps each) + the one-time ~1.5 GB download.
- Exp 3b (scale sweep): **~7–9 h on an M2 at the default R=5** — an overnight run (batch-64 is
  ~1 s/step and the heavy models 200–470 ms/step); `--online` adds ~30 s per tracing-ON run for the
  span flush (10 cells × R=5 ≈ +25 min). Use the `scale_sweep_min.yml` smoke (~10 min) first. To
  shorten the full run, lower `--runs` or the per-cell `max_batches` in `scale_sweep.yml` (each
  cell's CI is self-contained, so fewer steps only widens that cell's interval).

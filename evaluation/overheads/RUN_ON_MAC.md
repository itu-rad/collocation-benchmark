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
```

Drop `--online` to run fully local/offline (no credentials, for reproduction without server access).

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
```

See `framework_overhead/framework_overhead.md` and `modularity_overhead/modularity_overhead.md`
for the full write-ups and reference numbers.

## 5. Rough wall-times
- Exp 1 + 2: minutes to low-tens-of-minutes (15 depths × R × 2 arms; `--online` adds ~30 s per tracing-ON run for the flush).
- Exp 3: ~30–60 min on an M2 (R=5 × baseline + 2 arms, 1100 steps each) + the one-time ~1.5 GB download.

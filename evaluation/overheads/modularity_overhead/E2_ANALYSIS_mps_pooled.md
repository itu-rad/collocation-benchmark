# E2 — Modularity overhead (real workload, scale sweep)

Estimator: median of per-run paired differences (runs dropped as system warm-up: 1).
Warm-up steps dropped per run: 200. Metric: training-stage step (monotonic perf clock). Statistic: paired across-run difference (arms interleaved per repetition); 95% CI bootstrapped over run pairs.


# ===== mps (174 CSVs, 10 cells) =====

## mps — per-cell overhead (paired across runs)

| cell | R | step (ms) | core (µs/step, median) | core % | core (mean) | +tracing (µs/step) | total % |
|---|--:|--:|---|--:|--:|---|--:|
| ConvNeXt-L b8 | 3 | 193.91 | +482.9 [+86.1, +660.0] | +0.249% | +422.7 | +546.0 [+425.5, +797.3] | +0.282% |
| EfficientNetV2-L b8 | 3 | 460.54 | -575.8 [-737.8, -420.6] | -0.125% | -573.0 | -27.5 [-564.9, +264.0] | -0.006% |
| EfficientNetV2-M b8 | 3 | 247.31 | -88.7 [-205.1, +78.8] | -0.036% | -66.5 | +41.4 [-146.6, +181.0] | +0.017% |
| EfficientNetV2-S b1 | 7 | 29.30 | +78.3 [-126.3, +167.4] | +0.267% | +8.0 | +224.9 [+55.6, +440.0] | +0.768% |
| EfficientNetV2-S b2 | 5 | 29.79 | +255.1 [-759.3, +7615.1] | +0.857% | +273.7 | -379.0 [-1993.6, +3420.6] | -1.272% |
| EfficientNetV2-S b4 | 3 | 46.42 | +338.1 [-799.1, +820.1] | +0.728% | +327.4 | +631.9 [+483.5, +3181.8] | +1.361% |
| EfficientNetV2-S b8 | 9 | 89.35 | -6.0 [-58.8, +21.6] | -0.007% | -145.0 | -0.4 [-160.4, +250.7] | -0.000% |
| EfficientNetV2-S b16 | 3 | 178.16 | +539.5 [+398.0, +753.0] | +0.303% | +562.5 | +773.5 [+519.9, +976.2] | +0.434% |
| EfficientNetV2-S b32 | 3 | 348.84 | +143.2 [-958.6, +897.9] | +0.041% | +20.6 | +513.7 [+37.7, +769.8] | +0.147% |
| EfficientNetV2-S b64 | 3 | 690.75 | +625.5 [+155.5, +2583.5] | +0.091% | +1149.3 | +307.2 [-1490.7, +927.4] | +0.044% |

(core = Choreo wrapper, tracing off, vs the bare monolith; +tracing = wrapper + radt bulk/proc span export. Brackets: 95% CI, bootstrap over run pairs.)
- MIXED-REGIME repetitions dropped, EfficientNetV2-S b2: [12, 14] (arms landed in different step-time regimes)
- MIXED-REGIME repetitions dropped, EfficientNetV2-S b4: [11, 12, 13, 14] (arms landed in different step-time regimes)
- per-run paired core diffs (µs), ConvNeXt-L b8: +482.9 / +140.3 / +644.9
- per-run paired core diffs (µs), EfficientNetV2-L b8: -514.5 / -575.8 / -628.7
- per-run paired core diffs (µs), EfficientNetV2-M b8: -88.7 / -180.0 / +69.1
- per-run paired core diffs (µs), EfficientNetV2-S b1: -32.4 / +78.3 / +79.5 / +109.7 / -359.5 / +43.0 / +137.3
- per-run paired core diffs (µs), EfficientNetV2-S b2: -37.4 / +1179.6 / -289.4 / +255.1 / +260.4
- per-run paired core diffs (µs), EfficientNetV2-S b4: +614.3 / +338.1 / +29.7
- per-run paired core diffs (µs), EfficientNetV2-S b8: +7.6 / +30.0 / -50.6 / +21.5 / -6.0 / -1241.8 / -55.2 / +8.1 / -18.6
- per-run paired core diffs (µs), EfficientNetV2-S b16: +487.7 / +539.5 / +660.2
- per-run paired core diffs (µs), EfficientNetV2-S b32: -895.5 / +814.1 / +143.2
- per-run paired core diffs (µs), EfficientNetV2-S b64: +302.6 / +625.5 / +2519.9

### mps — Batch sweep (EfficientNetV2-S)

| batch | step (ms) | core µs/step | core % of step | total µs/step | total % |
|---|--:|--:|--:|--:|--:|
| 1 | 29.30 | +78.3 | +0.267% | +224.9 | +0.768% |
| 2 | 29.79 | +255.1 | +0.857% | -379.0 | -1.272% |
| 4 | 46.42 | +338.1 | +0.728% | +631.9 | +1.361% |
| 8 | 89.35 | -6.0 | -0.007% | -0.4 | -0.000% |
| 16 | 178.16 | +539.5 | +0.303% | +773.5 | +0.434% |
| 32 | 348.84 | +143.2 | +0.041% | +513.7 | +0.147% |
| 64 | 690.75 | +625.5 | +0.091% | +307.2 | +0.044% |

step 29.3 → 690.7 ms (23.6×): core +0.267% → +0.091% (relative shrinks), absolute +78.3 → +625.5 µs/step (spread 631.4 µs across cells; TOO NOISY to call fixed).

### mps — Model sweep (batch 8)

| model | step (ms) | core µs/step | core % of step | total µs/step | total % |
|---|--:|--:|--:|--:|--:|
| EfficientNetV2-S | 89.35 | -6.0 | -0.007% | -0.4 | -0.000% |
| ConvNeXt-L | 193.91 | +482.9 | +0.249% | +546.0 | +0.282% |
| EfficientNetV2-M | 247.31 | -88.7 | -0.036% | +41.4 | +0.017% |
| EfficientNetV2-L | 460.54 | -575.8 | -0.125% | -27.5 | -0.006% |

step 89.4 → 460.5 ms (5.2×): core -0.007% → -0.125% (relative does NOT shrink), absolute -6.0 → -575.8 µs/step (spread 1058.7 µs across cells; TOO NOISY to call fixed).

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e2_modularity_scale.png`


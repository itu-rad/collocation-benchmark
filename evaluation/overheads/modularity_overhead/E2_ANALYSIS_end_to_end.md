# E2 — Modularity overhead (real workload, scale sweep)

Estimator: median of per-run paired differences (runs dropped as system warm-up: 1).
Warm-up steps dropped per run: 200. Metric: training-stage step (monotonic perf clock). Statistic: paired across-run difference (arms interleaved per repetition); 95% CI bootstrapped over run pairs.


# ===== mps (162 CSVs, 9 cells) =====

## mps — per-cell overhead (paired across runs)

| cell | R | step (ms) | core (µs/step, median) | core % | core (mean) | +tracing (µs/step) | total % |
|---|--:|--:|---|--:|--:|---|--:|
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
- per-run paired core diffs (µs), EfficientNetV2-L b8: -514.5 / -575.8 / -628.7
- per-run paired core diffs (µs), EfficientNetV2-M b8: -88.7 / -180.0 / +69.1
- per-run paired core diffs (µs), EfficientNetV2-S b1: -32.4 / +78.3 / +79.5 / +109.7 / -359.5 / +43.0 / +137.3
- per-run paired core diffs (µs), EfficientNetV2-S b2: -37.4 / +1179.6 / -289.4 / +255.1 / +260.4
- per-run paired core diffs (µs), EfficientNetV2-S b4: +614.3 / +338.1 / +29.7
- per-run paired core diffs (µs), EfficientNetV2-S b8: +7.6 / +30.0 / -50.6 / +21.5 / -6.0 / -1241.8 / -55.2 / +8.1 / -18.6
- per-run paired core diffs (µs), EfficientNetV2-S b16: +487.7 / +539.5 / +660.2
- per-run paired core diffs (µs), EfficientNetV2-S b32: -895.5 / +814.1 / +143.2
- per-run paired core diffs (µs), EfficientNetV2-S b64: +302.6 / +625.5 / +2519.9

## mps — end-to-end cost (step PERIOD = 1/throughput)

| cell | R | period (ms) | step covers | core e2e (µs/step) | core e2e % | +tracing e2e (µs/step) | total e2e % |
|---|--:|--:|--:|---|--:|---|--:|
| EfficientNetV2-L b8 | 3 | 509.71 | 90.4% | -558.2 [-1312.8, -84.5] | -0.110% | +567.3 [-687.8, +1708.3] | +0.111% |
| EfficientNetV2-M b8 | 3 | 289.85 | 85.4% | -156.7 [-1262.8, +841.0] | -0.054% | +670.1 [-51.4, +1141.2] | +0.231% |
| EfficientNetV2-S b1 | 7 | 34.44 | 85.4% | +288.0 [-173.4, +484.8] | +0.836% | +828.0 [+341.5, +1094.0] | +2.404% |
| EfficientNetV2-S b2 | 5 | 39.17 | 72.6% | +4009.5 [-4728.7, +8735.1] | +10.236% | -1711.8 [-8951.1, +7822.7] | -4.370% |
| EfficientNetV2-S b4 | 3 | 66.22 | 74.1% | +603.3 [-122.8, +1504.4] | +0.911% | +3655.0 [+2472.8, +5873.7] | +5.519% |
| EfficientNetV2-S b8 | 9 | 127.44 | 70.2% | +102.7 [-892.7, +531.8] | +0.081% | +799.9 [-1342.8, +1564.1] | +0.628% |
| EfficientNetV2-S b16 | 3 | 245.55 | 72.7% | +263.2 [-647.4, +1412.6] | +0.107% | +1187.8 [-161.3, +2507.9] | +0.484% |
| EfficientNetV2-S b32 | 3 | 483.63 | 72.1% | -522.4 [-1560.0, +3125.0] | -0.108% | +1689.9 [+453.9, +5497.9] | +0.349% |
| EfficientNetV2-S b64 | 3 | 959.63 | 72.1% | -614.5 [-3373.0, +6001.5] | -0.064% | +1887.0 [-1080.3, +3678.2] | +0.197% |

(period = training-step start to the next step's start, on the same monotonic clock; `step covers` = median step duration / median period in the BASELINE arm, i.e. how much of the wall clock E2's headline metric can see at all. Brackets: 95% CI, bootstrap over run pairs.)

### mps — headline (in-step) vs end-to-end, side by side

| cell | core in-step µs | core in-step % | core e2e µs | core e2e % | understated by |
|---|--:|--:|--:|--:|--:|
| EfficientNetV2-L b8 | -575.8 | -0.125% | -558.2 | -0.110% | n/a |
| EfficientNetV2-M b8 | -88.7 | -0.036% | -156.7 | -0.054% | n/a |
| EfficientNetV2-S b1 | +78.3 | +0.267% | +288.0 | +0.836% | 3.7x |
| EfficientNetV2-S b2 | +255.1 | +0.857% | +4009.5 | +10.236% | 15.7x |
| EfficientNetV2-S b4 | +338.1 | +0.728% | +603.3 | +0.911% | 1.8x |
| EfficientNetV2-S b8 | -6.0 | -0.007% | +102.7 | +0.081% | n/a |
| EfficientNetV2-S b16 | +539.5 | +0.303% | +263.2 | +0.107% | 0.5x |
| EfficientNetV2-S b32 | +143.2 | +0.041% | -522.4 | -0.108% | n/a |
| EfficientNetV2-S b64 | +625.5 | +0.091% | -614.5 | -0.064% | n/a |

(`understated by` is only meaningful where BOTH estimates are positive; a negative estimate is apparatus noise, not a speed-up.)

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
| EfficientNetV2-M | 247.31 | -88.7 | -0.036% | +41.4 | +0.017% |
| EfficientNetV2-L | 460.54 | -575.8 | -0.125% | -27.5 | -0.006% |

step 89.4 → 460.5 ms (5.2×): core -0.007% → -0.125% (relative does NOT shrink), absolute -6.0 → -575.8 µs/step (spread 569.9 µs across cells; TOO NOISY to call fixed).

# ===== cuda (126 CSVs, 9 cells) =====

## cuda — per-cell overhead (paired across runs)

| cell | R | step (ms) | core (µs/step, median) | core % | core (mean) | +tracing (µs/step) | total % |
|---|--:|--:|---|--:|--:|---|--:|
| EfficientNetV2-L b8 | 3 | 189.96 | +640.8 [-833.3, +1183.2] | +0.337% | +330.0 | +271.7 [-166.5, +352.7] | +0.143% |
| EfficientNetV2-M b8 | 3 | 104.70 | -30.3 [-86.8, +514.6] | -0.029% | +133.6 | +40.8 [-48.1, +747.2] | +0.039% |
| EfficientNetV2-S b1 | 3 | 9.15 | +18.4 [-53.3, +152.2] | +0.201% | +45.5 | -203.6 [-370.2, +231.3] | -2.226% |
| EfficientNetV2-S b2 | 3 | 11.70 | -153.8 [-184.6, +1.5] | -1.314% | -108.7 | +23.8 [-39.3, +235.3] | +0.203% |
| EfficientNetV2-S b4 | 3 | 20.14 | +57.7 [+18.3, +149.8] | +0.287% | +81.5 | +177.8 [+141.3, +266.8] | +0.883% |
| EfficientNetV2-S b8 | 9 | 39.06 | +257.1 [-37.5, +310.4] | +0.658% | +153.3 | +264.0 [+117.9, +356.3] | +0.676% |
| EfficientNetV2-S b16 | 3 | 82.32 | +12.4 [-93.1, +87.3] | +0.015% | -4.3 | +111.8 [+1.9, +296.2] | +0.136% |
| EfficientNetV2-S b32 | 3 | 176.39 | -128.7 [-377.3, +67.0] | -0.073% | -155.6 | -273.3 [-424.1, +221.5] | -0.155% |
| EfficientNetV2-S b64 | 3 | 369.56 | +19.1 [-456.0, +270.0] | +0.005% | -55.0 | -189.5 [-699.7, +759.0] | -0.051% |

(core = Choreo wrapper, tracing off, vs the bare monolith; +tracing = wrapper + radt bulk/proc span export. Brackets: 95% CI, bootstrap over run pairs.)
- per-run paired core diffs (µs), EfficientNetV2-L b8: +640.8 / -801.5 / +1150.9
- per-run paired core diffs (µs), EfficientNetV2-M b8: -44.1 / +475.3 / -30.3
- per-run paired core diffs (µs), EfficientNetV2-S b1: -19.2 / +137.2 / +18.4
- per-run paired core diffs (µs), EfficientNetV2-S b2: -16.6 / -153.8 / -155.6
- per-run paired core diffs (µs), EfficientNetV2-S b4: +55.9 / +130.9 / +57.7
- per-run paired core diffs (µs), EfficientNetV2-S b8: +270.6 / +12.1 / +134.3 / -75.8 / +304.3 / +257.1 / -138.7 / +325.9 / +289.4
- per-run paired core diffs (µs), EfficientNetV2-S b16: +26.6 / +12.4 / -52.1
- per-run paired core diffs (µs), EfficientNetV2-S b32: -343.8 / -128.7 / +5.8
- per-run paired core diffs (µs), EfficientNetV2-S b64: -410.8 / +19.1 / +226.8

## cuda — end-to-end cost (step PERIOD = 1/throughput)

| cell | R | period (ms) | step covers | core e2e (µs/step) | core e2e % | +tracing e2e (µs/step) | total e2e % |
|---|--:|--:|--:|---|--:|---|--:|
| EfficientNetV2-L b8 | 3 | 224.43 | 84.7% | +2138.6 [+702.4, +3206.8] | +0.953% | +1964.6 [+1646.6, +2296.4] | +0.875% |
| EfficientNetV2-M b8 | 3 | 133.63 | 78.3% | +1669.8 [+1283.5, +2136.7] | +1.250% | +1861.8 [+1410.0, +2308.6] | +1.393% |
| EfficientNetV2-S b1 | 3 | 14.03 | 65.3% | +1763.3 [+1437.7, +1899.4] | +12.564% | +562.9 [+281.6, +1828.9] | +4.011% |
| EfficientNetV2-S b2 | 3 | 20.03 | 58.4% | +853.8 [+463.6, +1207.3] | +4.264% | +870.4 [+762.1, +1003.0] | +4.346% |
| EfficientNetV2-S b4 | 3 | 34.08 | 59.0% | +1600.2 [+1197.6, +1727.7] | +4.695% | +1492.4 [+1085.8, +1656.5] | +4.379% |
| EfficientNetV2-S b8 | 9 | 63.87 | 61.2% | +2075.9 [+1808.1, +2331.6] | +3.250% | +1925.7 [+1713.7, +2172.5] | +3.015% |
| EfficientNetV2-S b16 | 3 | 129.12 | 63.7% | +2076.1 [+1416.6, +2450.1] | +1.608% | +1871.0 [+1369.2, +2282.2] | +1.449% |
| EfficientNetV2-S b32 | 3 | 269.55 | 65.5% | +3163.3 [+2353.9, +4070.3] | +1.174% | +3503.2 [+2493.2, +4323.7] | +1.300% |
| EfficientNetV2-S b64 | 3 | 554.47 | 66.7% | +5847.9 [+4175.4, +8436.4] | +1.055% | +6960.9 [+4117.8, +9035.9] | +1.255% |

(period = training-step start to the next step's start, on the same monotonic clock; `step covers` = median step duration / median period in the BASELINE arm, i.e. how much of the wall clock E2's headline metric can see at all. Brackets: 95% CI, bootstrap over run pairs.)

### cuda — headline (in-step) vs end-to-end, side by side

| cell | core in-step µs | core in-step % | core e2e µs | core e2e % | understated by |
|---|--:|--:|--:|--:|--:|
| EfficientNetV2-L b8 | +640.8 | +0.337% | +2138.6 | +0.953% | 3.3x |
| EfficientNetV2-M b8 | -30.3 | -0.029% | +1669.8 | +1.250% | n/a |
| EfficientNetV2-S b1 | +18.4 | +0.201% | +1763.3 | +12.564% | 95.7x |
| EfficientNetV2-S b2 | -153.8 | -1.314% | +853.8 | +4.264% | n/a |
| EfficientNetV2-S b4 | +57.7 | +0.287% | +1600.2 | +4.695% | 27.7x |
| EfficientNetV2-S b8 | +257.1 | +0.658% | +2075.9 | +3.250% | 8.1x |
| EfficientNetV2-S b16 | +12.4 | +0.015% | +2076.1 | +1.608% | 167.3x |
| EfficientNetV2-S b32 | -128.7 | -0.073% | +3163.3 | +1.174% | n/a |
| EfficientNetV2-S b64 | +19.1 | +0.005% | +5847.9 | +1.055% | 306.8x |

(`understated by` is only meaningful where BOTH estimates are positive; a negative estimate is apparatus noise, not a speed-up.)

### cuda — Batch sweep (EfficientNetV2-S)

| batch | step (ms) | core µs/step | core % of step | total µs/step | total % |
|---|--:|--:|--:|--:|--:|
| 1 | 9.15 | +18.4 | +0.201% | -203.6 | -2.226% |
| 2 | 11.70 | -153.8 | -1.314% | +23.8 | +0.203% |
| 4 | 20.14 | +57.7 | +0.287% | +177.8 | +0.883% |
| 8 | 39.06 | +257.1 | +0.658% | +264.0 | +0.676% |
| 16 | 82.32 | +12.4 | +0.015% | +111.8 | +0.136% |
| 32 | 176.39 | -128.7 | -0.073% | -273.3 | -0.155% |
| 64 | 369.56 | +19.1 | +0.005% | -189.5 | -0.051% |

step 9.1 → 369.6 ms (40.4×): core +0.201% → +0.005% (relative shrinks), absolute +18.4 → +19.1 µs/step (spread 410.9 µs across cells; TOO NOISY to call fixed).

### cuda — Model sweep (batch 8)

| model | step (ms) | core µs/step | core % of step | total µs/step | total % |
|---|--:|--:|--:|--:|--:|
| EfficientNetV2-S | 39.06 | +257.1 | +0.658% | +264.0 | +0.676% |
| EfficientNetV2-M | 104.70 | -30.3 | -0.029% | +40.8 | +0.039% |
| EfficientNetV2-L | 189.96 | +640.8 | +0.337% | +271.7 | +0.143% |

step 39.1 → 190.0 ms (4.9×): core +0.658% → +0.337% (relative shrinks), absolute +257.1 → +640.8 µs/step (spread 671.1 µs across cells; TOO NOISY to call fixed).

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e2_modularity_scale.png`

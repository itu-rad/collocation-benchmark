# E3 — MLPerf / 3D-UNet: reproduction + the measurement boundary

Online serving regime: one request in flight (serialize_queries, queue_depth 1, batch 1). Latencies from the monotonic perf clock; per-case values are medians across repetitions.

- GB10 (cuda): 5 run(s), 42 cases per run
- M2 Pro (mps): 1 run(s), 42 cases per run

## Prong 1 — parity with the MLPerf reference harness (GB10, same device)

| quantity | MLPerf reference | Choreo | note |
|---|--:|--:|---|
| mean Dice (composite) | 0.8617 | — | same 42-case KiTS19 set |
| Dice kidney | 0.9347 | — | same 42-case KiTS19 set |
| Dice tumor | 0.7887 | — | same 42-case KiTS19 set |
| inference latency, mean (ms) | 8053 | 5837 | like-for-like: MLPerf times ONLY inference |
| inference latency, p90 (ms) | 16077 | 14413 | |
| end-to-end per request (ms) | not measured | 9178 | MLPerf's boundary excludes load+preprocess — prong 2 |

## Prong 2 — what MLPerf's measurement boundary hides (online serving)

| device | n cases | e2e median (ms) | load+preprocess (ms) | inference (ms) | preprocessing share | share range across cases |
|---|--:|--:|--:|--:|--:|---|
| GB10 (cuda) | 42 | 9171 | 1692 | 5833 | **17.4%** | 10.3–70.2% |
| M2 Pro (mps) | 42 | 70284 | 4179 | 63896 | **4.8%** | 1.7–24.2% |

The preprocessing share is what an offline-preload benchmark reports as zero. It cannot be hidden online: a request arrives with its own raw volume, so there is nothing to prefetch.

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e3_request_breakdown.png`

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e3_preprocessing_share.png`


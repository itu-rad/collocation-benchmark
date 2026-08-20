# E3 — MLPerf / 3D-UNet: reproduction + the measurement boundary

Online serving regime: one request in flight (serialize_queries, queue_depth 1, batch 1). Latencies from the monotonic perf clock; per-case values are medians across repetitions.

- GB10 (cuda): 5 run(s), 42 cases per run
- M2 Pro (mps): 2 run(s), 42 cases per run

## Prong 1 — parity with the MLPerf reference harness (GB10, same device)

| quantity | MLPerf reference | Choreo | note |
|---|--:|--:|---|
| mean Dice (composite) | 0.8617 | 0.8704 | Δ 0.0087 |
| Dice kidney | 0.9347 | 0.9484 | Δ 0.0137 |
| Dice tumor | 0.7887 | 0.7924 | Δ 0.0037 |
| inference latency, median (ms) | 6098 | 5837 | **4.3% apart** — like-for-like: MLPerf times ONLY inference |
| inference latency, mean (ms) | 8053 | 7848 | mean is case-mix sensitive; see median |
| inference latency, p90 (ms) | 16077 | 14413 | |
| end-to-end per request (ms) | not measured | 9178 | MLPerf's boundary excludes load+preprocess — prong 2 |

**Accuracy caveat — two differences, both known:** (a) the MLPerf reference postprocesses its logged predictions back to the ORIGINAL voxel spacing before scoring, while the Choreo number is scored on the resampled grid the model actually runs on; (b) the reference scores 43 cases while Choreo's inference_cases.json is a strict 42-case subset (it omits case_00400). So read the agreement as 'both clear the MLPerf accuracy gate (99% of 0.86170 = 0.8531)', not as a bit-exact match.

## Prong 2 — what MLPerf's measurement boundary hides (online serving)

| device | n cases | e2e median (ms) | load+preprocess (ms) | inference (ms) | preprocessing share | share range across cases |
|---|--:|--:|--:|--:|--:|---|
| GB10 (cuda) | 42 | 9171 | 1692 | 5833 | **17.4%** | 10.3–70.2% |
| M2 Pro (mps) | 42 | 69902 | 4078 | 63748 | **4.7%** | 1.6–23.8% |

The preprocessing share is what an offline-preload benchmark reports as zero. It cannot be hidden online: a request arrives with its own raw volume, so there is nothing to prefetch.

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e3_request_breakdown.png`

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e3_preprocessing_share.png`


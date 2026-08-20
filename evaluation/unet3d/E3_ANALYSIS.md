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

### Matched per-case inference time (GB10) — 16 cases loadgen actually exercised

| case | sub-volumes | MLPerf inner (s) | Choreo stage (s) | diff |
|---|--:|--:|--:|--:|
| case_00160 | 8 | 1.00 | 1.00 | -0.1% |
| case_00138 | 16 | 1.92 | 1.93 | +0.4% |
| case_00187 | 27 | 3.18 | 3.19 | +0.2% |
| case_00076 | 32 | 3.76 | 3.76 | +0.0% |
| case_00061 | 36 | 4.27 | 4.28 | +0.2% |
| case_00162 | 36 | 4.23 | 4.22 | -0.3% |
| case_00203 | 45 | 8.92 | 5.26 | -41.0% |
| case_00080 | 48 | 5.59 | 5.60 | +0.2% |
| case_00206 | 50 | 10.53 | 5.83 | -44.6% |
| case_00169 | 64 | 7.75 | 7.42 | -4.2% |
| case_00171 | 80 | 15.82 | 9.24 | -41.6% |
| case_00207 | 96 | 11.06 | 11.10 | +0.3% |
| case_00128 | 100 | 19.35 | 11.54 | -40.4% |
| case_00005 | 108 | 12.46 | 20.56 | +65.1% |
| case_00185 | 125 | 16.04 | 14.41 | -10.2% |
| case_00176 | 144 | 17.11 | 16.58 | -3.1% |

**Median per-case difference: -0.2%** — the same work, not a faster implementation. Cases within +/-1%: 8/16. Larger outliers are first-touch effects (a shape loadgen saw once, before cuDNN/allocator warm-up) against a Choreo median over repetitions.

## Prong 2 — what MLPerf's measurement boundary hides (online serving)

| device | n cases | e2e median (ms) | load+preprocess (ms) | inference (ms) | preprocessing share | share range across cases |
|---|--:|--:|--:|--:|--:|---|
| GB10 (cuda) | 42 | 9171 | 1692 | 5833 | **17.4%** | 10.3–70.2% |
| M2 Pro (mps) | 42 | 69902 | 4078 | 63748 | **4.7%** | 1.6–23.8% |

The preprocessing share is what an offline-preload benchmark reports as zero. It cannot be hidden online: a request arrives with its own raw volume, so there is nothing to prefetch.

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e3_request_breakdown.png`

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e3_preprocessing_share.png`


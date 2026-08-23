# E4 — Self-RAG decomposition: the prefill/decode split

prefill = first_token(start->end) (TTFT, compute-bound); decode = run_end - first_token_end (memory-bandwidth-bound). Pooled over LLM stages and repetitions; medians.

- M2 Pro (mlx): 32 run-file(s)
- GB10 (cuda): 56 run-file(s)

## M2 Pro (mlx) — prefill vs decode per arm

| task | arm | LLM calls | prefill median (ms) | decode median (ms) | unaccounted (ms) | prefill share of p+d | decode tok/s |
|---|---|--:|--:|--:|--:|--:|--:|
| factoid | decomposed | 118 | 7303 | 230 | 2 | 96.9% | 6.3 |
| factoid | decomposed_serial | 282 | 3193 | 51 | 104 | 98.4% | 20.4 |
| factoid | decomposed_shared | 356 | 2146 | 42 | 2952 | 98.1% | 24.3 |
| factoid | decomposed_shared_serial | 282 | 2734 | 42 | 3 | 98.5% | 24.1 |
| factoid | monolith | 152 | 4454 | 1177 | 0 | 79.1% | 27.2 |
| factoid | monolith_4b | 142 | 2601 | 502 | 0 | 83.8% | 46.0 |
| factoid | monolith_4b_serial | 96 | 3205 | 540 | 3 | 85.6% | 46.0 |
| factoid | monolith_serial | 102 | 5823 | 1252 | 4 | 82.3% | 27.0 |
| multihop | decomposed | 293 | 5184 | 162 | 3 | 97.0% | 13.9 |
| multihop | decomposed_serial | 312 | 2868 | 52 | 108 | 98.2% | 20.6 |
| multihop | decomposed_shared | 406 | 2212 | 41 | 0 | 98.2% | 24.7 |
| multihop | decomposed_shared_serial | 312 | 2228 | 42 | 2 | 98.1% | 24.2 |
| multihop | monolith | 308 | 4749 | 1095 | 0 | 81.3% | 27.2 |
| multihop | monolith_4b | 260 | 2712 | 459 | 0 | 85.5% | 45.9 |
| multihop | monolith_4b_serial | 198 | 2751 | 463 | 3 | 85.6% | 45.7 |
| multihop | monolith_serial | 252 | 4916 | 1105 | 4 | 81.7% | 26.9 |

## GB10 (cuda) — prefill vs decode per arm

| task | arm | LLM calls | prefill median (ms) | decode median (ms) | unaccounted (ms) | prefill share of p+d | decode tok/s |
|---|---|--:|--:|--:|--:|--:|--:|
| factoid | decomposed | 1176 | 487 | 247 | 3 | 66.3% | 17.2 |
| factoid | decomposed_serial | 376 | 480 | 142 | 3 | 77.2% | 22.0 |
| factoid | decomposed_shared | 1176 | 350 | 154 | 3 | 69.4% | 20.1 |
| factoid | decomposed_shared_serial | 376 | 473 | 142 | 3 | 76.9% | 21.8 |
| factoid | monolith | 390 | 554 | 1906 | 3 | 22.5% | 17.0 |
| factoid | monolith_4b | 402 | 422 | 1181 | 3 | 26.3% | 21.5 |
| factoid | monolith_4b_serial | 136 | 551 | 1192 | 3 | 31.6% | 22.1 |
| factoid | monolith_serial | 120 | 724 | 1916 | 3 | 27.4% | 16.9 |
| multihop | decomposed | 1280 | 468 | 249 | 3 | 65.3% | 18.9 |
| multihop | decomposed_serial | 464 | 401 | 145 | 3 | 73.5% | 22.0 |
| multihop | decomposed_shared | 1280 | 382 | 162 | 3 | 70.2% | 20.1 |
| multihop | decomposed_shared_serial | 464 | 398 | 142 | 3 | 73.6% | 22.3 |
| multihop | monolith | 794 | 596 | 1699 | 4 | 26.0% | 17.0 |
| multihop | monolith_4b | 852 | 438 | 1064 | 3 | 29.2% | 21.5 |
| multihop | monolith_4b_serial | 296 | 477 | 1034 | 3 | 31.6% | 22.2 |
| multihop | monolith_serial | 288 | 630 | 1649 | 3 | 27.7% | 17.5 |

### M2 Pro (mlx) — per-role phase shape

| arm | LLM role | calls | prefill median (ms) | decode median (ms) | prefill share | decode tok/s |
|---|---|--:|--:|--:|--:|--:|
| decomposed | Generator LLM | 74 | 9314 | 334 | 96.5% | 8.4 |
| decomposed | Grader LLM | 187 | 6083 | 89 | 98.6% | 11.8 |
| decomposed | Hallucination LLM | 72 | 9144 | 228 | 97.6% | 5.9 |
| decomposed | Query rewrite LLM | 78 | 2576 | 378 | 87.2% | 23.0 |
| decomposed_serial | Generator LLM | 168 | 2645 | 95 | 96.5% | 31.9 |
| decomposed_serial | Grader LLM | 219 | 19398 | 51 | 99.7% | 19.6 |
| decomposed_serial | Hallucination LLM | 168 | 2491 | 49 | 98.1% | 20.3 |
| decomposed_serial | Query rewrite LLM | 39 | 1933 | 160 | 92.3% | 38.6 |
| decomposed_shared | Shared LLM (generate) | 191 | 2136 | 83 | 96.3% | 35.5 |
| decomposed_shared | Shared LLM (grade) | 300 | 2370 | 41 | 98.3% | 24.5 |
| decomposed_shared | Shared LLM (hallucination) | 191 | 2057 | 41 | 98.0% | 24.4 |
| decomposed_shared | Shared LLM (rewrite) | 80 | 1892 | 209 | 90.1% | 42.8 |
| decomposed_shared_serial | Shared LLM (generate) | 168 | 2470 | 85 | 96.7% | 35.7 |
| decomposed_shared_serial | Shared LLM (grade) | 219 | 2543 | 42 | 98.4% | 23.8 |
| decomposed_shared_serial | Shared LLM (hallucination) | 168 | 2439 | 42 | 98.3% | 24.1 |
| decomposed_shared_serial | Shared LLM (rewrite) | 39 | 1896 | 150 | 92.7% | 40.7 |
| monolith | Monolith LLM | 340 | 5164 | 1116 | 82.2% | 27.3 |
| monolith | Query rewrite LLM | 120 | 3401 | 451 | 88.3% | 26.4 |
| monolith_4b | Monolith LLM | 311 | 2867 | 503 | 85.1% | 46.2 |
| monolith_4b | Query rewrite LLM | 91 | 1910 | 177 | 91.5% | 41.7 |
| monolith_4b_serial | Monolith LLM | 237 | 3116 | 510 | 85.9% | 46.0 |
| monolith_4b_serial | Query rewrite LLM | 57 | 1918 | 148 | 92.8% | 40.3 |
| monolith_serial | Monolith LLM | 267 | 5802 | 1172 | 83.2% | 27.1 |
| monolith_serial | Query rewrite LLM | 87 | 3455 | 387 | 89.9% | 26.0 |

### GB10 (cuda) — per-role phase shape

| arm | LLM role | calls | prefill median (ms) | decode median (ms) | prefill share | decode tok/s |
|---|---|--:|--:|--:|--:|--:|
| decomposed | Generator LLM | 613 | 544 | 347 | 61.1% | 14.2 |
| decomposed | Grader LLM | 945 | 483 | 161 | 74.9% | 18.6 |
| decomposed | Hallucination LLM | 613 | 455 | 189 | 70.7% | 15.9 |
| decomposed | Query rewrite LLM | 285 | 391 | 599 | 39.5% | 20.5 |
| decomposed_serial | Generator LLM | 212 | 443 | 237 | 65.1% | 22.2 |
| decomposed_serial | Grader LLM | 328 | 478 | 137 | 77.8% | 22.0 |
| decomposed_serial | Hallucination LLM | 212 | 422 | 137 | 75.5% | 21.9 |
| decomposed_serial | Query rewrite LLM | 88 | 352 | 408 | 46.3% | 22.1 |
| decomposed_shared | Shared LLM (generate) | 613 | 367 | 211 | 63.5% | 20.5 |
| decomposed_shared | Shared LLM (grade) | 945 | 403 | 153 | 72.4% | 19.6 |
| decomposed_shared | Shared LLM (hallucination) | 613 | 355 | 151 | 70.2% | 19.9 |
| decomposed_shared | Shared LLM (rewrite) | 285 | 335 | 496 | 40.3% | 21.5 |
| decomposed_shared_serial | Shared LLM (generate) | 212 | 443 | 239 | 65.0% | 22.1 |
| decomposed_shared_serial | Shared LLM (grade) | 328 | 471 | 136 | 77.6% | 22.0 |
| decomposed_shared_serial | Shared LLM (hallucination) | 212 | 423 | 136 | 75.7% | 22.0 |
| decomposed_shared_serial | Shared LLM (rewrite) | 88 | 370 | 377 | 49.6% | 22.9 |
| monolith | Monolith LLM | 922 | 615 | 1845 | 25.0% | 17.1 |
| monolith | Query rewrite LLM | 262 | 457 | 665 | 40.8% | 16.7 |
| monolith_4b | Monolith LLM | 957 | 475 | 1159 | 29.0% | 21.5 |
| monolith_4b | Query rewrite LLM | 297 | 334 | 459 | 42.1% | 21.5 |
| monolith_4b_serial | Monolith LLM | 336 | 543 | 1145 | 32.2% | 22.2 |
| monolith_4b_serial | Query rewrite LLM | 96 | 348 | 351 | 49.8% | 22.3 |
| monolith_serial | Monolith LLM | 324 | 718 | 1835 | 28.1% | 17.3 |
| monolith_serial | Query rewrite LLM | 84 | 500 | 563 | 47.0% | 17.3 |

## Cross-device: prefill/decode balance (M2 Pro (mlx) vs GB10 (cuda))

| task | arm | prefill mlx | prefill cuda | prefill speedup | tok/s mlx | tok/s cuda | decode speedup (per token) | decode ratio (per call) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| factoid | decomposed | 7303 | 487 | **15.00x** | 6.3 | 17.2 | **2.71x** | 0.93x |
| factoid | decomposed_serial | 3193 | 480 | **6.65x** | 20.4 | 22.0 | **1.08x** | 0.36x |
| factoid | decomposed_shared | 2146 | 350 | **6.13x** | 24.3 | 20.1 | **0.83x** | 0.27x |
| factoid | decomposed_shared_serial | 2734 | 473 | **5.78x** | 24.1 | 21.8 | **0.90x** | 0.30x |
| factoid | monolith | 4454 | 554 | **8.04x** | 27.2 | 17.0 | **0.62x** | 0.62x |
| factoid | monolith_4b | 2601 | 422 | **6.16x** | 46.0 | 21.5 | **0.47x** | 0.43x |
| factoid | monolith_4b_serial | 3205 | 551 | **5.82x** | 46.0 | 22.1 | **0.48x** | 0.45x |
| factoid | monolith_serial | 5823 | 724 | **8.04x** | 27.0 | 16.9 | **0.63x** | 0.65x |
| multihop | decomposed | 5184 | 468 | **11.07x** | 13.9 | 18.9 | **1.36x** | 0.65x |
| multihop | decomposed_serial | 2868 | 401 | **7.15x** | 20.6 | 22.0 | **1.07x** | 0.36x |
| multihop | decomposed_shared | 2212 | 382 | **5.79x** | 24.7 | 20.1 | **0.81x** | 0.25x |
| multihop | decomposed_shared_serial | 2228 | 398 | **5.60x** | 24.2 | 22.3 | **0.92x** | 0.30x |
| multihop | monolith | 4749 | 596 | **7.97x** | 27.2 | 17.0 | **0.62x** | 0.64x |
| multihop | monolith_4b | 2712 | 438 | **6.19x** | 45.9 | 21.5 | **0.47x** | 0.43x |
| multihop | monolith_4b_serial | 2751 | 477 | **5.76x** | 45.7 | 22.2 | **0.49x** | 0.45x |
| multihop | monolith_serial | 4916 | 630 | **7.80x** | 26.9 | 17.5 | **0.65x** | 0.67x |

**Read the per-TOKEN column, not the per-call one.** Decode duration is
(tokens x time-per-token), and the two backends do NOT emit the same number of
tokens for the same prompt under greedy decoding — MLX 4-bit and BitsAndBytes NF4
produce different outputs, e.g. decomposed_shared emits ~2.0 tokens/call on mlx
against ~4.0 on cuda. A per-call decode ratio therefore mixes decode SPEED with
how long the model chose to talk, and understates cuda by up to 2x. tok/s is the
speed measure; the per-call column is retained only to show the size of that
distortion.

If the compute-bound prefill speeds up much more than the memory-bound decode
across devices, the phase balance shifts — which is what moves the optimal
decomposition and can flip which arm wins.

## Cross-device with 95% CIs — M2 Pro (mlx) vs GB10 (cuda) (run 1 dropped as warm-up)

| task | arm | prefill speedup [95% CI] | decode speedup [95% CI] | runs |
|---|---|--:|--:|--:|
| factoid | decomposed | **6.63x** [6.54, 6.89] | 0.36x [0.36, 0.36] | 2 |
| factoid | decomposed_shared | **5.78x** [5.63, 5.96] | 0.30x [0.29, 0.30] | 2 |
| factoid | monolith_4b | **5.82x** [5.64, 6.05] | 0.45x [0.44, 0.45] | 2 |
| factoid | monolith | **7.99x** [7.57, 8.16] | 0.65x [0.64, 0.66] | 2 |
| multihop | decomposed | **7.17x** [6.83, 7.25] | 0.36x [0.35, 0.36] | 2 |
| multihop | decomposed_shared | **5.61x** [5.45, 5.72] | 0.29x [0.29, 0.30] | 2 |
| multihop | monolith_4b | **5.76x** [5.57, 5.81] | 0.45x [0.44, 0.46] | 2 |
| multihop | monolith | **7.85x** [7.71, 7.98] | 0.67x [0.66, 0.68] | 2 |

Decode here is a per-CALL duration ratio; see the tok/s columns for the per-token rate, which is the speed measure.

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e4_prefill_decode.png`


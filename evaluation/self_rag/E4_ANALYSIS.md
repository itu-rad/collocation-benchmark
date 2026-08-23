# E4 — Self-RAG decomposition: the prefill/decode split

prefill = first_token(start->end) (TTFT, compute-bound); decode = run_end - first_token_end (memory-bandwidth-bound). Pooled over LLM stages and repetitions; medians.

- M2 Pro (mlx): 8 run-file(s)
- GB10 (cuda): 24 run-file(s)

## M2 Pro (mlx) — prefill vs decode per arm

| task | arm | queries | prefill median (ms) | decode median (ms) | prefill share | decode tok/s |
|---|---|--:|--:|--:|--:|--:|
| factoid | decomposed | 118 | 7303 | 230 | 96.9% | 11.5 |
| factoid | decomposed_shared | 356 | 2146 | 42 | 98.1% | 48.3 |
| factoid | monolith | 152 | 4454 | 1177 | 79.1% | 28.1 |
| factoid | monolith_4b | 142 | 2601 | 502 | 83.8% | 48.1 |
| multihop | decomposed | 293 | 5184 | 162 | 97.0% | 24.3 |
| multihop | decomposed_shared | 406 | 2212 | 41 | 98.2% | 49.0 |
| multihop | monolith | 308 | 4749 | 1095 | 81.3% | 28.3 |
| multihop | monolith_4b | 260 | 2712 | 459 | 85.5% | 48.1 |

## GB10 (cuda) — prefill vs decode per arm

| task | arm | queries | prefill median (ms) | decode median (ms) | prefill share | decode tok/s |
|---|---|--:|--:|--:|--:|--:|
| factoid | decomposed | 1176 | 487 | 247 | 66.3% | 21.8 |
| factoid | decomposed_shared | 1176 | 350 | 154 | 69.4% | 26.1 |
| factoid | monolith | 390 | 554 | 1906 | 22.5% | 17.6 |
| factoid | monolith_4b | 402 | 422 | 1181 | 26.3% | 22.3 |
| multihop | decomposed | 1280 | 468 | 249 | 65.3% | 23.7 |
| multihop | decomposed_shared | 1280 | 382 | 162 | 70.2% | 25.3 |
| multihop | monolith | 794 | 596 | 1699 | 26.0% | 17.8 |
| multihop | monolith_4b | 852 | 438 | 1064 | 29.2% | 22.7 |

### M2 Pro (mlx) — per-role phase shape

| arm | LLM role | calls | prefill median (ms) | decode median (ms) | prefill share | decode tok/s |
|---|---|--:|--:|--:|--:|--:|
| decomposed | Generator LLM | 74 | 9314 | 334 | 96.5% | 12.1 |
| decomposed | Grader LLM | 187 | 6083 | 89 | 98.6% | 22.6 |
| decomposed | Hallucination LLM | 72 | 9144 | 228 | 97.6% | 11.1 |
| decomposed | Query rewrite LLM | 78 | 2576 | 378 | 87.2% | 25.8 |
| decomposed_shared | Shared LLM (generate) | 191 | 2136 | 83 | 96.3% | 48.1 |
| decomposed_shared | Shared LLM (grade) | 300 | 2370 | 41 | 98.3% | 48.9 |
| decomposed_shared | Shared LLM (hallucination) | 191 | 2057 | 41 | 98.0% | 48.7 |
| decomposed_shared | Shared LLM (rewrite) | 80 | 1892 | 209 | 90.1% | 48.0 |
| monolith | Monolith LLM | 340 | 5164 | 1116 | 82.2% | 28.2 |
| monolith | Query rewrite LLM | 120 | 3401 | 451 | 88.3% | 28.7 |
| monolith_4b | Monolith LLM | 311 | 2867 | 503 | 85.1% | 48.1 |
| monolith_4b | Query rewrite LLM | 91 | 1910 | 177 | 91.5% | 47.8 |

### GB10 (cuda) — per-role phase shape

| arm | LLM role | calls | prefill median (ms) | decode median (ms) | prefill share | decode tok/s |
|---|---|--:|--:|--:|--:|--:|
| decomposed | Generator LLM | 613 | 544 | 347 | 61.1% | 16.9 |
| decomposed | Grader LLM | 945 | 483 | 161 | 74.9% | 24.8 |
| decomposed | Hallucination LLM | 613 | 455 | 189 | 70.7% | 21.2 |
| decomposed | Query rewrite LLM | 285 | 391 | 599 | 39.5% | 22.7 |
| decomposed_shared | Shared LLM (generate) | 613 | 367 | 211 | 63.5% | 25.2 |
| decomposed_shared | Shared LLM (grade) | 945 | 403 | 153 | 72.4% | 26.1 |
| decomposed_shared | Shared LLM (hallucination) | 613 | 355 | 151 | 70.2% | 26.5 |
| decomposed_shared | Shared LLM (rewrite) | 285 | 335 | 496 | 40.3% | 23.6 |
| monolith | Monolith LLM | 922 | 615 | 1845 | 25.0% | 17.6 |
| monolith | Query rewrite LLM | 262 | 457 | 665 | 40.8% | 18.2 |
| monolith_4b | Monolith LLM | 957 | 475 | 1159 | 29.0% | 22.4 |
| monolith_4b | Query rewrite LLM | 297 | 334 | 459 | 42.1% | 23.8 |

## Cross-device: prefill/decode balance (M2 Pro (mlx) vs GB10 (cuda))

| task | arm | prefill mlx | prefill cuda | prefill speedup | tok/s mlx | tok/s cuda | decode speedup (per token) | decode ratio (per call) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| factoid | decomposed | 7303 | 487 | **15.00x** | 11.5 | 21.8 | **1.89x** | 0.93x |
| factoid | decomposed_shared | 2146 | 350 | **6.13x** | 48.3 | 26.1 | **0.54x** | 0.27x |
| factoid | monolith | 4454 | 554 | **8.04x** | 28.1 | 17.6 | **0.62x** | 0.62x |
| factoid | monolith_4b | 2601 | 422 | **6.16x** | 48.1 | 22.3 | **0.47x** | 0.43x |
| multihop | decomposed | 5184 | 468 | **11.07x** | 24.3 | 23.7 | **0.98x** | 0.65x |
| multihop | decomposed_shared | 2212 | 382 | **5.79x** | 49.0 | 25.3 | **0.52x** | 0.25x |
| multihop | monolith | 4749 | 596 | **7.97x** | 28.3 | 17.8 | **0.63x** | 0.64x |
| multihop | monolith_4b | 2712 | 438 | **6.19x** | 48.1 | 22.7 | **0.47x** | 0.43x |

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

**Figure:** `/Users/roba/Documents/work/research/collocation-benchmark/evaluation/overheads/paper_assets/e4_prefill_decode.png`


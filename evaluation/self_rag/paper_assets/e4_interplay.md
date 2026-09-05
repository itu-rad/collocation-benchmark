# 5.1 — the interplay table

Latency from the listener-off serial pass; power and memory from the listener-on counter pass; quality from the LLM judge over the same 30 questions on both machines.

## m3pro  (power: SoC package)

| task | strategy | quality | prefill (ms) | decode (ms) | decode tok/s | power (W) | peak mem (GB) |
|---|---|--:|--:|--:|--:|--:|--:|
| factoid | monolith | 0.867 | 3457 | 1625 | 20.9 | 19.3 | 17.1 |
| factoid | monolith_4b | 0.933 | 1902 | 695 | 35.9 | 18.8 | 14.1 |
| factoid | decomposed | 0.867 | 1660 | 51 | 19.8 | 19.7 | 17.7 |
| factoid | decomposed_shared | 0.867 | 1655 | 50 | 20.0 | 21.2 | 14.1 |
| multihop | monolith | 0.433 | 2925 | 1439 | 20.8 | 20.3 | 17.1 |
| multihop | monolith_4b | 0.333 | 1655 | 590 | 35.6 | 19.8 | 13.9 |
| multihop | decomposed | 0.433 | 1360 | 51 | 19.7 | 20.1 | 17.6 |
| multihop | decomposed_shared | 0.433 | 1339 | 51 | 19.8 | 20.5 | 14.2 |

## gb10  (power: GPU only)

| task | strategy | quality | prefill (ms) | decode (ms) | decode tok/s | power (W) | peak mem (GB) | energy (J) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| factoid | monolith | 0.933 | 724 | 1916 | 16.9 | 28.9 | 40.0 | 5136.8 |
| factoid | monolith_4b | 0.900 | 551 | 1192 | 22.1 | 28.5 | 21.7 | 3611.6 |
| factoid | decomposed | 0.867 | 480 | 142 | 22.0 | 23.1 | 28.6 | 5119.1 |
| factoid | decomposed_shared | 0.867 | 473 | 142 | 21.8 | 31.3 | 22.2 | 4130.2 |
| multihop | monolith | 0.300 | 630 | 1649 | 17.5 | 35.4 | 39.8 | 8944.7 |
| multihop | monolith_4b | 0.433 | 477 | 1034 | 22.2 | 33.0 | 21.8 | 5857.4 |
| multihop | decomposed | 0.433 | 401 | 145 | 22.0 | 25.6 | 28.1 | 6121.4 |
| multihop | decomposed_shared | 0.433 | 398 | 142 | 22.3 | 33.5 | 21.9 | 5111.8 |

Power is not comparable across machines: macmon reports SoC package power, DCGM reports GPU power only and so excludes the CPU side. Energy is a DCGM cumulative counter, reported here as the per-run delta in joules, and exists on gb10 only. Peak memory is system memory on both -- meaningful on unified-memory parts, but not GPU framebuffer, and on the shared gb10 it includes other tenants.

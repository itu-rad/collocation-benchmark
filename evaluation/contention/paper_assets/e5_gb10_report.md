# 5.2 — collocation types on gb10

Run is the unit of replication; quantiles pooled across runs, 95% CI by hierarchical (run-then-query) bootstrap. Repetition 1 dropped as warm-up.

uncontended baseline: p50 673 ms, p95 2241 ms  (5 runs, 500 queries)

| cell | runs | fg p50 (ms) | 95% CI | fg p95 (ms) | vs baseline p50 | fg tput (q/s) | bg tput (q/s) |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 5 | 673 | [638, 706] | 2241 |  | 0.18 | — |
| bg_cpu | 5 | 701 | [666, 732] | 2281 | +4% | 0.17 | 4.87 |
| bg_gpu_mps | 5 | 979 | [895, 1059] | 3849 | +46% | 0.16 | 13.80 |
| bg_gpu_timesliced | 5 | 1005 | [952, 1055] | 3724 | +49% | 0.18 | 13.80 |

**Figure:** `evaluation/contention/paper_assets/e5_gb10_degradation.png`

bg tput is the background pipeline's own delivered rate, from its own run —
the attribution this arrangement exists to provide. '—' means the cell has
no background (the baseline).

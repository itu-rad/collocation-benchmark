# 5.2 — collocation types on m3pro

Run is the unit of replication; quantiles pooled across runs, 95% CI by hierarchical (run-then-query) bootstrap. Repetition 1 dropped as warm-up.

uncontended baseline: p50 998 ms, p95 1657 ms  (5 runs, 400 queries)

| cell | runs | fg p50 (ms) | 95% CI | fg p95 (ms) | vs baseline p50 | fg tput (q/s) | bg tput (q/s) |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 5 | 998 | [972, 1050] | 1657 |  | 0.10 | — |
| bg_ane | 5 | 1021 | [991, 1052] | 1759 | +2% | 0.09 | 9.19 |
| bg_cpu | 5 | 1015 | [972, 1053] | 1779 | +2% | 0.10 | 1.97 |
| bg_gpu | 5 | 1101 | [1068, 1135] | 2093 | +10% | 0.11 | 3.45 |
| dose_GB12 | 2 | 1027 | [978, 1087] | 1898 | +3% | 0.11 | 1.97 |
| dose_GB16 | 2 | 1011 | [973, 1078] | 1711 | +1% | 0.09 | 2.63 |
| dose_GB4 | 2 | 988 | [931, 1044] | 1690 | -1% | 0.09 | 0.66 |
| dose_GB8 | 2 | 1001 | [967, 1073] | 1719 | +0% | 0.10 | 1.32 |

**Figure:** `evaluation/contention/paper_assets/e5_m3pro_degradation.png`

bg tput is the background pipeline's own delivered rate, from its own run —
the attribution this arrangement exists to provide. '—' means the cell has
no background (the baseline).

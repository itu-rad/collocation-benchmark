# Paired quality power analysis (decision 5: N=120 vs N=200)

Device `mlx`. Arms paired on normalized question text (query_id is per-run, not arm-stable). EM via McNemar mid-p; ΔEM/ΔF1 via paired question-level bootstrap (10000 resamples, seed 1234). Equivalence margins: EM ±0.05, F1 ±0.05. `N_needed` = questions to shrink the paired-diff 95% half-width to the margin.

> **R=1 sufficiency for quality**: greedy decoding (`do_sample=false`) is empirically deterministic on this setup — two independent runs of the factoid monolith cell (2026-07-14 vs 2026-07-20, distinct query_ids) produced **byte-identical answers on all 120 questions**. So EM/F1 have ~zero *between-run* variance and the only variance is *within-run* (question-level), which this bootstrap fully captures. `N_needed` is therefore the ACTUAL required sample size, not a lower bound, and R=1 is sufficient for the quality claim. (Latency/throughput metrics DO retain run variance — thermal/scheduling — so that R=1 caveat still applies to the performance results, just not to accuracy.)

## factoid: monolith vs decomposed

- N (shared questions): **120**
- EM: monolith=0.467  decomposed=0.575  ΔEM=+0.108  95% CI [+0.033, +0.183]
  - McNemar discordant b=6 c=19, mid-p=0.009 → EM DIFFERENCE detected
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.445; half-width now=0.080, @N=200=0.062; N_needed for ±0.05 = **305**
- F1: monolith=0.514  decomposed=0.653  ΔF1=+0.139  95% CI [+0.066, +0.215]
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.427; half-width now=0.076, @N=200=0.059; N_needed for ±0.05 = **280**

## factoid: monolith vs decomposed_shared

- N (shared questions): **120**
- EM: monolith=0.467  decomposed_shared=0.575  ΔEM=+0.108  95% CI [+0.033, +0.183]
  - McNemar discordant b=6 c=19, mid-p=0.009 → EM DIFFERENCE detected
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.445; half-width now=0.080, @N=200=0.062; N_needed for ±0.05 = **305**
- F1: monolith=0.514  decomposed_shared=0.653  ΔF1=+0.139  95% CI [+0.066, +0.215]
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.427; half-width now=0.076, @N=200=0.059; N_needed for ±0.05 = **280**

## factoid: monolith vs monolith_4b

- N (shared questions): **120**
- EM: monolith=0.467  monolith_4b=0.575  ΔEM=+0.108  95% CI [+0.033, +0.183]
  - McNemar discordant b=5 c=18, mid-p=0.007 → EM DIFFERENCE detected
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.426; half-width now=0.076, @N=200=0.059; N_needed for ±0.05 = **279**
- F1: monolith=0.514  monolith_4b=0.634  ΔF1=+0.120  95% CI [+0.047, +0.195]
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.420; half-width now=0.075, @N=200=0.058; N_needed for ±0.05 = **271**

## multihop: monolith vs decomposed

- N (shared questions): **120**
- EM: monolith=0.217  decomposed=0.267  ΔEM=+0.050  95% CI [-0.008, +0.108]
  - McNemar discordant b=4 c=10, mid-p=0.118 → no detectable EM difference
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.339; half-width now=0.061, @N=200=0.047; N_needed for ±0.05 = **177**
- F1: monolith=0.269  decomposed=0.335  ΔF1=+0.065  95% CI [+0.004, +0.127]
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.353; half-width now=0.063, @N=200=0.049; N_needed for ±0.05 = **192**

## multihop: monolith vs decomposed_shared

- N (shared questions): **120**
- EM: monolith=0.217  decomposed_shared=0.267  ΔEM=+0.050  95% CI [-0.008, +0.108]
  - McNemar discordant b=4 c=10, mid-p=0.118 → no detectable EM difference
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.339; half-width now=0.061, @N=200=0.047; N_needed for ±0.05 = **177**
- F1: monolith=0.269  decomposed_shared=0.335  ΔF1=+0.065  95% CI [+0.004, +0.127]
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.353; half-width now=0.063, @N=200=0.049; N_needed for ±0.05 = **192**

## multihop: monolith vs monolith_4b

- N (shared questions): **120**
- EM: monolith=0.217  monolith_4b=0.242  ΔEM=+0.025  95% CI [-0.025, +0.075]
  - McNemar discordant b=4 c=7, mid-p=0.388 → no detectable EM difference
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.303; half-width now=0.054, @N=200=0.042; N_needed for ±0.05 = **142**
- F1: monolith=0.269  monolith_4b=0.319  ΔF1=+0.049  95% CI [-0.009, +0.108]
  - equivalence @±0.05: **NOT certified** at N=120
  - paired-diff SD=0.329; half-width now=0.059, @N=200=0.046; N_needed for ±0.05 = **167**

## Verdict

- Not all contrasts certify at N=120. Largest N required across contrasts/metrics = **305**. Even N=200 is insufficient — see per-contrast N_needed.

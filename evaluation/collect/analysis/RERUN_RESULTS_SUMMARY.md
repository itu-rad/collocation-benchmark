# AMC-fixed R=1 re-run — results summary (mlx / M2 Pro 16GB)

Full re-collection after the AMC bandwidth-counter fix. 42 cells collected + analyzed;
e7_rung_27b hit the 16GB OOM ceiling (run failed, log idle 92 min) — itself the E7 finding.

## AMC FIX VERIFIED (the #1 architect-reviewer blocker)
All 31/31 staged bandwidth cells peak 168.7–184.8 GB/s (median 170.3), ALL ≤200 GB/s
(physically valid on the M2 Pro LPDDR5 bus), vs the archived-broken 369 GB/s (all >200,
impossible). The fix holds across every stage type (a/b/c-clipane/c-clipgpu/c-stream/d).

## STAGED CONTENTION — engine-specific contention (analyze_staged.py; point estimates, R=1)
Stage C (fg median response vs co-runner AMC bandwidth), normalized dose-response slopes:
  clipANE 9.30e-11  >  clipGPU 4.08e-11  (ratio 2.28×)  >>  stream/CPU −1.9e-12 (negligible)
=> an ANE co-runner degrades the foreground LLM ~2.3× more per unit memory bandwidth than a
GPU co-runner; a CPU co-runner barely matters. Supports the "engine-specific contention" reframe.
Stage A/B (system view): fg p50 2.01→2.26 s, fg throughput −12% as bg intensity rises.
CAVEAT: R=1 → every slope-ratio/H1/H2 CI is degenerate (zero-width); directional only. Need R≥2
for Fieller/bootstrap ratio intervals before a band verdict.

## QUALITY — decomposition (quality_power.py; McNemar + power)
Factoid: monolith vs decomposed ΔEM +0.108 [0.033,0.183] ΔF1 +0.139 p=0.009 (SIG);
         monolith_4b matches (+0.108). Multihop: ΔEM +0.050 [−0.008,0.108] p=0.118 (near-null).
Consistent with the committed Section B (decomposition helps the easy task, null on the hard one).

## E7 CAPACITY LADDER — factoid EM vs model size (4-bit, 16GB)
  0.8B → EM 0.450 | 2B → 0.492 | 4B → 0.500 | 27B → OOM CEILING (does not fit)
Quality scales with size to ~4B (diminishing returns), then a hard memory wall at 27B.

## DATA QUALITY (validate_pass.py): 0 PASS / 43 WARN / 15 FAIL
- 15 FAIL: R-QDEPTH — tiny transient queue blocks (1–3 blocked puts, max 7–14 ms) on staged
  cells; minor coordinated-omission caveat (magnitude ~ms out of ~500 s runs).
- 43 WARN: p95 gate unreachable at R=1 (pooled queries < 500; even at planned R=5) — paper must
  raise max_queries/R or pre-register dropping p95 for the staged experiments.
- e7_rung_27b: run failed (OOM ceiling). e7_rung_2b: 6/120 empty (retry-exhaustion; answered-rate
  is itself a metric).

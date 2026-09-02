# E1 — Framework overhead (NoOp microbenchmark)

**Question.** Before any end-to-end number can be trusted, what does the framework itself cost?
Specifically: is the per-stage cost (a) small, (b) flat in graph depth rather than compounding,
and (c) O(1) in the payload handed between stages rather than O(bytes)?

If all three hold, overhead seen in a case study is attributable to the workload, not to us.

## What is run

Chains of N pass-through stages that do no work, so the only thing measured is the framework's
dispatch: thread wake + queue hand-off. Two configurations, named for the role each plays rather
than the switch that sets it — they are the only two ways the framework is actually run:

| configuration | stage CSV rows | spans | what it is |
|---|---|---|---|
| `uninstrumented` | off | off | the framework bare |
| `+ tracing` | off | on | the framework as it is actually run |

`+ tracing − uninstrumented` is therefore what tracing costs. Sweeps: **depth** {1,2,4,…,128}
at zero payload, and **payload** {0, 1 KiB, 1 MiB, 10 MiB} × {ref, copy} at depth 10.

## Results (R=11, run 1 dropped; hierarchical bootstrap, run = unit of replication)

**Per-stage cost is small and amortizes with depth** — the marginal slope of L_q vs depth:

| machine | uninstrumented | + tracing | fixed per-query intercept (uninstr.) |
|---|--:|--:|--:|
| m3pro | **12.03 µs/stage** | 42.14 µs/stage | 25.63 µs |
| gb10 | **9.37 µs/stage** | 25.31 µs/stage | 15.95 µs |

Both fits are R² ≥ 0.996 over 8 depths, i.e. the cost is genuinely linear in depth with no
compounding term. Per-stage O(d) = L_q/depth falls from ~41 µs at depth 1 to ~12 µs at depth 128
on m3pro — the fixed per-query cost amortizing, not a per-stage cost that grows.

**Payload passing is zero-copy — O(1) in bytes, and the contrast is the evidence:**

| mode | cost vs payload | reading |
|---|--:|---|
| `ref` (ours) | **−0.006 µs/MB** | flat — passing a reference |
| `copy` | **66.4 µs/MB** | linear in payload — what a deep copy would cost |

At 10 MiB that is the difference between a constant ~31 µs and ~700 µs per stage. The `copy` arm
exists solely to show the measurement can detect an O(payload) term if one were there.

## Why this matters for the rest of the evaluation

These numbers are what license every later claim. E2 confirms they survive on a real workload
(framework term 194–562 µs/query, falling to 0.03–0.10% of a query as it gets heavier). Where a
case study reports a per-stage or per-phase breakdown, this is the floor beneath it.

## Reproduce

```bash
bash evaluation/overheads/framework_overhead/collect_e1.sh m3pro 11 143
PIN=19 bash evaluation/overheads/framework_overhead/collect_e1.sh gb10 11 143   # one X925 core
python evaluation/overheads/framework_overhead/analyze_e1.py
```

`E1_PAYLOAD=1` collects the payload sweep instead of the depth sweep. Results land in
`results/<machine>/`; figures in `paper_assets/`. Local MLflow store, not res17 — the documented
exemption for the overhead experiments, which emit spans at a rate no real workload approaches.

## Caveats to state

- **gb10 must be pinned to one X925 performance core** (`PIN=19`). Unpinned, the Grace scheduler
  migrates the run between core types and the timing goes bimodal — spreads of 29.8–393.2 µs
  collapse to 0.5–18.9 µs once pinned.
- `serialize_queries` is not used here; the NoOp chain measures dispatch, not scheduling.
- The depth-32 `+ tracing` cell on gb10 has a reproducible ~11 µs spread that is per-run sticky
  and remains unexplained. It does not move the fitted slope.

# §5.2 — Collocation types and per-pipeline attribution

**Status: not yet collected.** Framing agreed; all cells are new. Authoritative:
`../../EXPERIMENTS.md` → *E5 / §5.2*.

**Question.** When two pipelines share a node, what is the foreground *actually contending on*
under each collocation type — the compute engine, the shared memory bandwidth, or nothing the
partitioning mechanism can reach — and can the interference be attributed to the pipeline and
resource causing it? Degradation numbers are the symptom; the attribution is the deliverable.

**Subject: the collocation type, not the background workload.** The indexer and the
MemoryStream antagonist are props. The capability on show is that every number is attributed
to its pipeline — each runs in its own process with its own radt run, spans and listeners — so
a contention study knows who paid what. This is a build-up on §5.1: the foreground is the same
Self-RAG serving pipeline.

## The collocation axis

| machine | types |
|---|---|
| m3pro | background on GPU / **ANE** / CPU — three engines, one unified-memory pool |
| gb10 | time-sliced GPU / **MPS** / CPU |

**MPS only in the text.** MIG gets a half-day verification and, as expected on this part, one
line: *MIG is not supported on this device.* radt configures MPS automatically, which answers by
demonstration the gap the paper's own related work names — that configuring MPS/MIG "requires
significant manual effort and expertise".

## Exhibits

1. **Per-pipeline attribution table** — foreground p50/p95 and throughput per collocation type,
   background throughput alongside.
2. **Separation ≠ isolation, counter-confirmed** (claim and proof merged): moving the background
   to ANE or CPU removes compute contention; if degradation persists, unified-memory bandwidth is
   the shared resource. Proof by dose–response — the MemoryStream antagonist at stepped
   intensities, foreground latency against offered bandwidth, with **AMC per-engine DRAM counters
   confirming the dose on m3pro**. This is the paper's flagship "a hardware counter explains an
   application symptom" exhibit.
3. **What MPS partitioning buys** versus time-slicing when the memory system stays shared.

## Device asymmetry — state it, do not paper over it

| | gb10 | m3pro |
|---|---|---|
| power / energy | DCGM power + cumulative energy | macmon power / thermals |
| DRAM bandwidth | **none** — DCGM profiling fields confirmed unavailable on this stack | **AMC per-engine rd/wr bytes** (real counters, 2 Hz) |

So the bandwidth claim is directly measurable only on the Mac; on gb10 it is a stated
power/utilization proxy. Known caveat: under pure-GPU load roughly half the Mac's traffic lands
in "other" AMC agents rather than the GPU bucket.

## Grid

Collocation types at one calibrated intensity everywhere; dose–response on m3pro only, where the
counters carry the narrative; gb10 types-only, run overnight behind the occupancy gate with arm
order rotated.

## Machinery

`analyze_staged.py`, `staged_lib.py`, `generate_stage_configs.py`, `amc_calibration.py` here;
the AMC sampler under `../../scripts/`. Hardware facts and the calibration record live in
`../../CONTENTION_EXPERIMENTS_REDESIGN.md` and `AMC_CALIBRATION.md`.

**Note:** `analyze_staged.py`'s docstring still names `evaluation/collect/results/` as its
default input. That tree was removed as superseded; the path is repointed when the §5.2 harness
lands.

## Blocking before collection

Listeners are off on every serial config, so the profiling contribution has no supporting data
at all. That is the gate — see the §5 plan.

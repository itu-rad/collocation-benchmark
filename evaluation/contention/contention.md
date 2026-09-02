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

## Hardware attribution — what each device can actually measure

*(Consolidated from the retired `CONTENTION_EXPERIMENTS_REDESIGN.md`; verified 2026-07-13/14.)*

**m3pro — counter-backed.** `powermetrics` has no bandwidth sampler on this build and `macmon`
exposes only `ram_power`. The decisive channel is IOReport's **"AMC Stats / Perf Counters"**:
per-requestor DRAM *byte* counters with separate RD/WR channels for PCPU/ECPU, **GFX (GPU)**,
**ANE**, AVD, DISP — readable **without root** at ≥1 Hz. So we get not just total bandwidth but
**per-engine attribution**, i.e. we can report *who* moved the bytes.

Sampler: `scripts/amc_bandwidth_sampler.c` (+ `.py` wrapper, auto-compiles, writes
`<label>_bandwidth.csv` and offers an `AMCBandwidthSampler` context manager). Timestamps are
wall-clock, matching the trace's `%(created)f` for joins. Validated: CPU bucket 1.8 → 24.7 GB/s
under a ~19.5 GB/s induced stream; GPU bucket 0.8 → 70 GB/s under an MPS matmul loop (total
~146 GB/s against a ~200 GB/s roof); idle ≈7 GB/s aggregate DCS.

Three calibration facts that must survive into the analysis:
1. **Sum only per-requestor DCS RD/WR channels.** Summing all AMC agents double-counts fabric
   hops (ATC/SB/AFI).
2. **The agent→engine map is incomplete.** Under pure-GPU load ~half the traffic reports under
   AFR*/other agents, and kernel page-zero traffic from CPU allocations lands outside the
   PCPU/ECPU channels. Derive the mapping empirically with one single-engine calibration load
   per engine (CPU stream / GPU matmul / ANE encode) before collection.
3. **ABI hazard**, documented in both C files: `IOReportSimpleGetIntegerValue`'s second argument
   must be NULL — passing a local address corrupts the caller's stack.

**gb10 — proxy-backed.** `nvidia-smi utilization.memory` (memory-controller busy fraction) is
the proxy of record. DCGM `PROF_DRAM_ACTIVE` is **confirmed unavailable**: nv-hostengine runs as
a systemd service but its Profiling module reports "Failed to load" on this GB10 stack (checked
2026-07-14; not a privileges issue — closed). No usable Grace uncore events via unprivileged
`perf` either. DCGM power and cumulative energy *do* work and are what gb10 contributes.

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
the AMC sampler under `../../scripts/`. The calibration record is in `AMC_CALIBRATION.md`; the hardware-attribution facts are
consolidated above.

**Note:** `analyze_staged.py`'s docstring still names `evaluation/collect/results/` as its
default input. That tree was removed as superseded; the path is repointed when the §5.2 harness
lands.

## Blocking before collection

Listeners are off on every serial config, so the profiling contribution has no supporting data
at all. That is the gate — see the §5 plan.

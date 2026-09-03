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

## The M3 Pro does not populate the per-engine DRAM counters (2026-09-03)

The per-requestor AMC byte counters that exhibit 2 rests on **work on the M2 Pro and not on the
M3 Pro** — the machine §5.1 and the rest of the paper report. On m3pro all 48
`AMC Stats / Perf Counters / * DCS RD|WR` channels are present and every one reads **zero**, under
heavy memory load as well as idle. (Two sampler bugs had to be fixed before this was visible at
all: an M3-only subscription failure, and aggregate-vs-requestor being keyed on a `DIE` prefix
that M3 does not use. Both are fixed; the counters are still dead.)

What *does* respond on m3pro is the memory-controller energy channel,
`Energy Model / AMCC`, and it responds strongly and repeatably:

| m3pro | AMCC energy |
|---|---|
| idle | 153, 151, 155 mJ/s |
| heavy memory traffic | 1553, 1528, 1527 mJ/s |

≈10× dynamic range, tight across repeats. It is a real hardware counter reacting to the memory
system — but it is a **single aggregate**, so it can confirm a dose without attributing it to an
engine.

**Author decision needed.** Three ways to keep exhibit 2 honest:

1. **Dose-response on the M2 Pro**, where per-engine bytes work. Keeps full attribution
   (cpu/gpu/**ane** read+write); costs a third machine in the paper, stated plainly. The exhibit is
   a mechanism demonstration, so the machine may differ from §5.1's as long as it is named.
2. **AMCC energy on m3pro.** Stays on the paper's machine and still counter-explains the symptom,
   but in energy rather than bytes, and without the per-engine split that made the claim
   distinctive.
3. **Both** — AMCC for the dose on m3pro, per-engine bytes on the M2 Pro for the attribution.

Recommendation: **(1)**, because "separation is not isolation" needs the per-engine split to be
more than an assertion — showing the ANE background still moving DRAM bytes *is* the finding, and
an aggregate energy curve cannot show it.

## Device asymmetry — state it, do not paper over it

| | gb10 | m3pro |
|---|---|---|
| power / energy | DCGM power + cumulative energy | macmon power / thermals |
| DRAM bandwidth | **none** — DCGM profiling fields confirmed unavailable on this stack | **none on M3 Pro** — the AMC channels exist but read zero; per-engine rd/wr bytes work on the M2 Pro only. `Energy Model / AMCC` responds on M3 (aggregate, ~10x range) |

So the bandwidth claim is directly measurable only on the Mac; on gb10 it is a stated
power/utilization proxy. Known caveat: under pure-GPU load roughly half the Mac's traffic lands
in "other" AMC agents rather than the GPU bucket.

## First data (2026-09-03)

The harness is validated end-to-end on gb10. The **uncontended foreground baseline** — the
reference every degradation number is measured against — is:

| | gb10, foreground alone, 100 Poisson queries |
|---|---|
| p50 | 703 ms |
| p95 | 2643 ms |
| p99 / max | 5537 ms |
| counters | DCGMI + TOP, 1.00 Hz, full run window |

Per-query latencies come from the `pipeline - val` rows of the foreground trace (pipeline-level
run start/end, one pair per query); the whole-run `pipeline` rows are the process bookends, not
per-query, which is worth knowing before writing the analyzer.

## Prerequisites, per machine

Checked 2026-09-02. Missing prerequisites are silent at run time -- a missing model surfaces
as a failed import several minutes into a run, which is how the m3pro half of §5.1 went a month
without anyone noticing it had never run there.

| | m3pro | gb10 |
|---|---|---|
| foreground generator (Qwen3.5-4B) | present | present |
| `openai/clip-vit-large-patch14` | **missing** — downloads on first use | present |
| `tmp/clip_vit_l14_vision.mlpackage` (ANE arm) | **missing** — must be exported | n/a |
| `coremltools` | 9.0 | n/a |

The CoreML package is a hard prerequisite for the ANE cell, which carries exhibit 2. Build it
with:

    python stages/multimodal_vqa/export_clip_coreml.py \
        --model openai/clip-vit-large-patch14 \
        --output tmp/clip_vit_l14_vision.mlpackage

Do not run the export while a collection is in flight: it is CPU/ANE-heavy and would contend
with exactly what the collection measures.

## Grid

Collocation types at one calibrated intensity everywhere; dose–response on m3pro only, where the
counters carry the narrative; gb10 types-only, run overnight behind the occupancy gate with arm
order rotated.

## One YAML, one run, one process per pipeline

A cell is a single config passed to `main.py`. The config declares both pipelines — the RAG-serve
foreground from §5.1 and a single-resource background co-runner — and `main.py`'s orchestrator
mode (no `-p`) builds radt's schedule in memory with **one row per pipeline**, which radt then
launches as **separate processes** (`main.py <cfg> -p <n>`).

That process separation is what the section needs, and it was there all along: each pipeline gets
its own radt run, its own listeners and its own spans, so every number is attributed to the
pipeline that caused it — and MPS, which partitions between processes rather than threads, has
something to partition. Verified on gb10: two GPU contexts, 8192 MiB foreground and 1913 MiB
background, from one config.

*(An earlier note here claimed the fused configs ran in one process and had to be split. That was
wrong — it was an artefact of the harnesses invoking `main.py ... -p 0`, which selects a single
pipeline and bypasses the schedule entirely. The split configs and their generator were removed.)*

## Asking for a collocation mechanism

`collocation:` is a key in the same pipeline YAML as `listeners:`:

| value | meaning |
|---|---|
| `""` (default) | time-sliced — the processes share the GPU as usual |
| `"mps"` | radt brings up the CUDA MPS control daemon for the group |
| `"1g.10gb"` etc. | a MIG profile string |

`main.py` forwards it into the schedule row radt already builds, and radt's own `make_mps()` does
the setup. So **the claim that radt configures MPS is demonstrated literally**, no CSV and no
second mechanism — and the time-sliced and MPS cells are the same config differing in one line,
which is the point the section makes. `generate_stage_configs.py` emits the MPS twin alongside
each GPU cell.

## Machinery

`collect_e5.sh` here, alongside `analyze_staged.py`, `staged_lib.py`,
`generate_stage_configs.py` (which now also writes each device's listeners into every config,
sizes each background to outlast the foreground, and emits the MPS twins) and `amc_calibration.py`;
the AMC sampler under `../../scripts/`. The calibration record is in `AMC_CALIBRATION.md`; the hardware-attribution facts are
consolidated above.

**`analyze_staged.py` does not read the current data.** It implements the superseded staged
design and discovers cells by globbing `stage_*`, whereas `collect_e5.sh` labels runs
`e5_<cell>_{fg,bg}_<machine>_r<N>`. Its dangling paths (the removed `evaluation/collect/results/`
and `evaluation/results/` trees) are repointed at `results/<machine>/`, but porting the discovery
and the per-role split to the two-process labels is the remaining §5.2 analysis work.

It is kept rather than deleted because its statistics are what exhibit 2 needs and are shared
with the rest of the paper through `staged_lib`: hierarchical run-then-query bootstrap CIs,
dose–response slopes, and the matched-bytes/s slope-ratio comparison.

## Blocking before collection

**Cleared (2026-09-03).** Listeners record on both machines, and the harness proves it per pass
rather than assuming: `collect_e5.sh` aborts if the first cell lands no `system/*` series on the
tracking server. Observer cost is measured, not feared — +0.0% on m3pro (macmon), −0.7% on gb10
(dcgmi+top); see `../self_rag/self_rag.md`.

Two prerequisites bit here and are now checked by `scripts/radt_gate.py`: radt **patch 0002**
(without it the multi-pipeline schedule path hangs before any workload starts — no children, no
GPU work, no output) and a stale `radtlock` left by a killed schedule.

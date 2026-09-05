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
exposes only `ram_power`. The decisive channels are IOReport's, read **without root** at ≥1 Hz,
and which family works depends on the SoC generation (see "resolved" below):

- **M2-family:** **"AMC Stats / Perf Counters"** — per-requestor DRAM *byte* counters with
  separate RD/WR channels for PCPU/ECPU, **GFX (GPU)**, **ANE**, AVD, DISP. Exact.
- **M3-family (m3pro, the reported machine):** **"PMP / DCS BW"** — per-requestor DRAM
  *bandwidth histograms* for EACC0/PACC0, **AGX (GPU)**, **ANE0**, ISP, DISP. Bytes are derived.
  The AMC group exists here but its driver refuses the subscription.

Either way we get not just total bandwidth but **per-engine attribution**, i.e. we can report
*who* moved the bytes.

Sampler: `scripts/amc_bandwidth_sampler.c` (+ `.py` wrapper, auto-compiles, writes
`<label>_bandwidth.csv` and offers an `AMCBandwidthSampler` context manager). Timestamps are
wall-clock, matching the trace's `%(created)f` for joins. Validated **on the M2 Pro** (amc
backend): CPU bucket 1.8 → 24.7 GB/s under a ~19.5 GB/s induced stream; GPU bucket 0.8 →
70 GB/s under an MPS matmul loop (total ~146 GB/s against a ~200 GB/s roof); idle ≈7 GB/s
aggregate DCS. Validated **on m3pro** (pmp backend) separately — see "resolved" below.

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

## Per-engine DRAM counters on the M3 Pro: resolved (2026-09-05)

An earlier version of this section (2026-09-03) concluded that m3pro "does not populate" the
per-requestor DRAM counters, and asked for an author decision between running exhibit 2 on the
M2 Pro, falling back to aggregate `Energy Model / AMCC`, or both. **That diagnosis was wrong and
the decision is moot: per-engine DRAM attribution works on m3pro, unprivileged, at ≥1 Hz —
including the ANE.** Exhibit 2 stays on the machine the rest of the paper reports.

**What was actually wrong.** The AMC channels do not "read zero" — they never reach a sample at
all. On m3pro `AppleH15MemCacheController` refuses the IOReport subscription outright (a single
channel fails, so it is not a size or naming problem), and when the request is built from the
all-channel dictionary the library returns success and drops the AMC channels silently. A
per-group census settles it: AMC Stats is the **only** group on the machine that is copied and
subscribed but never sampled.

| stage | M2 Pro | m3pro |
|---|--:|--:|
| copied by `IOReportCopyAllChannels` | 117 | 128 |
| present in `subbedChannels` | 117 | 128 |
| **present in a sample** | **117** | **0** |

The sampler's guard iterated the *sample* looking for AMC channels and found none, then reported
"present but all read zero" — which is what the earlier reading rested on.

**The replacement.** m3pro publishes per-requestor DRAM bandwidth under the **PMP** group instead:
`PMP / DCS BW / {EACC0, PACC0, AGX, ANE0, ISP, DISP, ...} {RD, WR, RD+WR}`, from
`RTBuddyIOReportingEndpoint`, which subscribes and samples normally. `DCS` is the same
DRAM-command-scheduler layer the M2's counters measure. These are `kIOReportFormatState` 32-bin
*bandwidth histograms*, not byte counters, which is why tools reading them with
`IOReportSimpleGetIntegerValue` saw nothing. `scripts/amc_bandwidth_sampler.c` now selects the
backend automatically from what actually samples, and emits the same byte-valued CSV either way.

**Validated on m3pro** (2026-09-05, full tables in `AMC_CALIBRATION.md`): a paced known-byte CPU
read stream reads back at 0.94–1.04× of delivered across 3–30 GB/s; an ANE workload puts
**19.9 GB/s** on the `ane` bucket with the GPU provably idle, corroborated by `ANE / IOP State`
(Running 100%) and `Energy Model / ANE` (6591 mJ vs 0 at idle). Attribution is clean in every
case: the loaded engine moves, the others sit at their idle baseline.

**Two limits that must be stated wherever an m3pro bytes/s number appears.** Per-requestor
histograms saturate above **32 GB/s** and the aggregate above **64 GB/s** (the top bin is a
catch-all), so a saturating foreground pegs its bucket — the CSV's `saturated` column flags any
such row, and those rows are lower bounds. And bin 0 is a `<1 GB/s` catch-all, so sub-1 GB/s
traffic is indistinguishable from idle. Neither binds the flagship claim: the ANE sits at
~20 GB/s, comfortably inside the range.

**Collection consequence — the ANE needs warming up.** CoreML does not engage the ANE for the
first **~8 s** of a run; it serves early predictions on the CPU while compiling for the ANE, and
`ANE0` reports *no ticks at all* during that window even though throughput looks healthy. An ANE
background sampled cold reads as "the ANE moved nothing" — the exact false negative that would
sink the exhibit. Allow ≥10 s of warm-up before the measurement window opens. Relatedly, the
repo's exported `tmp/clip_vit_l14_vision.mlpackage` never reaches the ANE at all (`IOP State` =
Off throughout); it is currently a CPU workload and must be fixed before it can serve as the ANE
background.

## Cross-validating the M3 backend (2026-09-05)

The M3 Pro reads per-engine DRAM traffic through the PMP histogram backend rather than the M2's
exact byte counters. Checked independently before trusting it for the section:

| check | result |
|---|---|
| idle, machine clear | 4.97 GB/s total, 3.06 CPU — comparable to the M2 Pro's 2–4 |
| unthrottled memory copy | CPU 3.06 → 43.4 GB/s — attribution tracks |
| **at the GB12 operating point** | **CPU 11.82 GB/s against the 12.0 target** derived from the M2 Pro's exact counters |

That last row is the one that matters: a different machine, through a different instrument, agrees
within 1.5% with a rate derived from a different machine's exact byte counters. The matched-bytes
axis holds across both.

**The `saturated` column means peaks, not a bad average.** 30 of 39 rows flag saturated at the
GB12 point, because the memory-stream co-runner is bursty at fine grain — a 256 MB sweep at full
memory speed, then idle until the next arrival — so instantaneous ticks land in the top bin while
the interval average is 12 GB/s. Discarding saturated rows, as a literal reading of the flag would
have you do, throws away nearly all valid data. Treat it as "this row contains peaks above the
bin range", and judge the average on whether it matches the offered dose, which here it does.

**A caveat that cost an hour.** The first idle reading was 40 GB/s, which is not idle. An orphaned
workload from a killed collection was still running at 122% CPU. Check `ps -eo pcpu,etime,args -r`
before trusting any bandwidth baseline on this machine — and note it runs a GUI session, so
`WindowServer` is a permanent ~40% background load that a headless DUT would not have.

## gb10 result (R=6, repetition 1 dropped; complete 2026-09-05)

| cell | fg p50 | 95% CI | fg p95 | vs baseline p50 | bg throughput |
|---|--:|---|--:|--:|--:|
| baseline (alone) | 673 ms | [638, 706] | 2241 ms | — | — |
| CPU background | 701 ms | [666, 732] | 2281 ms | +4% | 4.87 q/s |
| GPU, MPS | 979 ms | [895, 1059] | 3849 ms | +46% | 13.80 q/s |
| GPU, time-sliced | 1005 ms | [952, 1055] | 3724 ms | +49% | 13.80 q/s |

**MPS does not measurably beat time-slicing here.** The p50 confidence intervals overlap across
almost their whole range ([895, 1059] against [952, 1055]), and at p95 MPS is nominally *worse*
(3849 vs 3724) with intervals overlapping even more. The background delivers the same 13.80 q/s
either way, so this is not MPS buying foreground latency at the background's expense — it is
partitioning making no difference to a workload pair this size.

That is a negative result and the section should report it as one. It is also a caution about
reading early data: at R=3 the same cells read 948 vs 1017 ms and looked like a 7% MPS win, which
did not survive to R=5.

**A CPU background costs the GPU foreground almost nothing** (+4%, and its interval overlaps the
baseline's), against +46-49% for a GPU background. On gb10 the compute engine, not the memory
system, dominates — which is the opposite of what the unified-memory Apple part is expected to
show, and is why the m3pro half carries the memory argument.

## Engine attribution is not pipeline attribution

The AMC/PMP sampler is machine-wide: it says which ENGINE moved bytes, not which pipeline. On
m3pro the foreground is an MLX model that also runs on the GPU, so in the GPU cell the `gpu`
bucket mixes foreground and background traffic (measured r1: gpu 9.7, cpu 2.4, ane 0.00, total
14.6 GB/s).

Pipeline attribution comes from the other half of the design — each pipeline is its own process,
its own radt run, its own listeners and spans. The two are combined, not conflated.

**This is why the ANE cell carries exhibit 2.** With the background on the ANE and the foreground
on the GPU, the two attributions coincide: `ane_rd + ane_wr` is unambiguously the background, and
nothing else on the machine is using that engine. The `ane = 0.00` reading in the GPU cell above
is the control that makes it readable.

## Which machine carries which claim

The intensity axis is matched on **bytes/s**, which m3pro can calibrate and gb10 cannot — gb10 has
no DRAM counter at all. Rather than fake a match there, each machine carries the claim its
instruments support.

| | m3pro | gb10 |
|---|---|---|
| cross-engine (GPU / ANE / CPU) | **matched at 12 GB/s**, per-engine power from macmon | not instrumentable |
| partitioning (time-sliced vs MPS) | n/a (no MPS) | **matched by construction** — same config, one line differs |
| dose–response | GB4/8/12/16 ladder | n/a |

**Why gb10 is not proxy-matched.** The only candidate was DCGM's memory-copy utilization, and it
is GPU-side: measured 2026-09-05, a CPU-side memory stream moving heavy DRAM traffic registers
**0%** on it, against 0% idle. A proxy blind to one of the two co-runners cannot match them.
Borrowing the Apple bytes/query would be worse still — different framework, precision and memory
architecture.

So gb10's contribution is the partitioning comparison, which needs no intensity decision because
both cells run the *same* config and differ only in `collocation:`. Its CPU cell is kept as a
supporting datum at a capacity-matched level and is **labelled as not bytes-matched**: it can show
whether degradation survives moving the background off the GPU, not why.

## Device asymmetry — state it, do not paper over it

| | gb10 | m3pro |
|---|---|---|
| power / energy | DCGM power + cumulative energy | macmon power / thermals |
| DRAM bandwidth | **none** — DCGM profiling fields confirmed unavailable on this stack | **per-engine rd/wr, no root** — via `PMP / DCS BW` bandwidth histograms (the AMC byte counters are unsubscribable on M3-family; the sampler switches backend automatically). Derived bytes, accurate below a 32 GB/s per-requestor ceiling |

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

**`analyze_e5.py` is the section's analyzer.** It reads what `collect_e5.sh` writes — one file
per pipeline per run — and produces the per-pipeline attribution table: foreground p50/p95 with a
hierarchical (run-then-query) bootstrap CI, degradation against the uncontended baseline, and the
background's own delivered throughput from its own run. Per-query latency comes from the
`pipeline - <split>` rows; the bare `pipeline` rows are process bookends and using them yields two
"latencies" per run.

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

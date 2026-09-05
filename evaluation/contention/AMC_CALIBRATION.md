# Per-engine DRAM counter calibration — M2 Pro (amc) and M3 Pro (pmp)

Two backends, one per SoC generation; `scripts/amc_bandwidth_sampler.c` picks between
them automatically. §1 is the M2 Pro (exact byte counters), §2 the M3 Pro (derived
from bandwidth histograms). See `docs/amc-m3-counters-plan.md` for why they differ.

## 1. M2 Pro — `amc` backend (byte counters)

Run: `python evaluation/contention/amc_calibration.py` (idle machine, AC power, no root).
Recorded **2026-08-31**, commit `c971469`.

## Why this exists

The mock-ASPLOS review raised it as a disqualifying blocker (`REVIEW_SYNTHESIS.md` B1): the
bandwidth axis reported **331 GB/s on a ~200 GB/s bus**, which is physically impossible, so
no bytes/s claim about the M2 Pro could stand.

**That reading predates the fix.** Commit `43d33a7` (2026-07-20) found the sampler summing
the memory-controller *aggregate* channel (`DCS RD/WR`, no requestor prefix) **and** its
per-requestor components — exactly double-counting. Half of 331 is ~165 GB/s, which sits
under spec and matches what a saturating GPU load actually delivers (see below). The same
commit added this calibration protocol; it had never been run.

## Method

Drive a load whose byte count is unambiguous and compare it to what the counters report.
`torch.dot` / `np.dot` over two 1-D operands far larger than any cache: both are read
exactly once per call, nothing is written back, so there is no write-allocate guesswork.
Each phase is differenced against an idle baseline sampled in the same session.

- CPU phase: 2 × 768 MB float64 (numpy, BLAS-threaded)
- GPU phase: 2 × 192 MB float32 on `mps`, with `torch.mps.synchronize()` so the elapsed time
  bounds work *retired on the device*, not merely enqueued

## Result — closes on both buckets

| bucket | known delivered | AMC delta | **factor (known/AMC)** | attribution | residual |
|---|--:|--:|--:|---|--:|
| CPU | 60.8 GB/s | 60.9 GB/s | **0.999** | cpu=100% | — |
| GPU | 162.2 GB/s | 161.4 GB/s | **1.005** | gpu=100% | **+0.0%** |

Peak observed total **168.7 GB/s**, i.e. 84% of the ~200 GB/s LPDDR5 ceiling — a plausible
achievable fraction, and no longer above it.

**Agent → bucket map** (from `amc_bandwidth_sampler.c`): `PCPU*`/`ECPU*` → `cpu`,
`GFX` → `gpu`, `ANE`/`ANS` → `ane`, everything else → `other`. The bare `DCS RD/WR` channel
is the controller aggregate and is reported as the total; it must **not** be summed with the
per-requestor channels.

## What this does and does not license

**Does:** the M2 Pro bytes/s axis is trustworthy to within ~0.6% for CPU-driven and
GPU-driven DRAM reads, with clean per-requestor attribution and no unattributed residual.

**Does not:** this is a *read-only* closure test. Write traffic and mixed read/write loads
are not calibrated, and write-allocate behaviour is exactly where a counter axis is most
likely to mislead. The ANE bucket is untested (no known-byte ANE load was driven). Both
should be closed before any claim that leans on write bandwidth or on ANE attribution.

Re-run this on any machine before reporting its bytes/s, and after any change to
`amc_bandwidth_sampler.c`.

## 2. M3 Pro — `pmp` backend (bandwidth histograms)

Recorded **2026-09-05** on `itu-mac` (Apple M3 Pro, 18 GB, macOS 26.6.2), no root.

M3-family silicon cannot use the counters in §1 at all: `AppleH15MemCacheController` refuses the
IOReport subscription, so the `AMC Stats` channels are copied and subscribed but **never reach a
sample** (0 of 128, the only such group on the machine). The sampler falls back to
`PMP / DCS BW / <REQ> {RD,WR}` — `kIOReportFormatState` 32-bin histograms where bin *i* carries
its upper edge in GB/s and its residency is PMP ticks spent in that bin. Bytes are therefore
**derived**, not counted: `mean_GBps × dt`, with each bucket normalised against the never-gated
aggregate channel's tick count.

### Method

Same principle as §1 — drive a load whose byte count is unambiguous — but paced across a range,
because a histogram can be right at one point and wrong at another.

- CPU phase: read-only sum over a 2 GB buffer (reads exactly once, writes nothing), busy-wait
  paced so the instantaneous rate equals the target rather than bursting.
- GPU phase: `torch.dot` on `mps` with `torch.mps.synchronize()`, paced.
- ANE phase: a deliberately ANE-friendly fp16 conv stack (`scripts/make_ane_model.py`,
  24 convs × 16.8 MB activations), driven by `scripts/ane_load.py`.

### CPU — closes across the range

| delivered (GB/s) | `PACC0 RD` reported | ratio |
|--:|--:|--:|
| 3.00 | 3.13 | 1.04 |
| 6.00 | 5.77 | 0.96 |
| 12.00 | 11.21 | 0.93 |
| 18.00 | 16.97 | 0.94 |
| 25.00 | 23.44 | 0.94 |
| 30.00 | 28.10 | 0.94 |

Slope ≈ 0.94. The ~6 % deficit is the right direction for a DRAM-side counter: reads served by
the system-level cache never reach DCS. Through the sampler's `cpu` bucket (both clusters) a
20.00 GB/s stream reads back as 19.4 GB/s, **ratio 0.97**.

### ANE — the bucket §1 left untested, now closed

| | `ANE0 RD+WR` | `AGX RD+WR` | corroboration |
|---|--:|--:|---|
| `CPU_ONLY` (6.1 pred/s) | *no ticks* | 0.50 (idle) | `ANE / IOP State` = Off 100 % |
| `CPU_AND_NE` (31.2 pred/s) | **19.75** | 0.50 (idle) | `IOP State` = **Running 100 %**; `Energy Model / ANE` = **6591 mJ** vs 0 idle |

Through the sampler: `ane` = **19.94 GB/s** (10.0 rd + 9.9 wr), `ane_duty` 1.00, `saturated` 0,
GPU at 0.07. Two independent channels confirm the placement, so ANE attribution is no longer an
assertion.

### GPU — attribution exact, absolute value caveated

| delivered (GB/s) | `AGX RD` reported | `PACC0 RD` |
|--:|--:|--:|
| 5.00 | 11.07 | 0.58 |
| 11.99 | 17.12 | 0.59 |
| 19.99 | 21.03 | 0.61 |
| 128.87 | 30.70 (pegged) | 0.56 |

`PACC0` never leaves its idle baseline while `AGX` tracks, so *which engine* is never in doubt.
But the absolute value **over-reports for bursty loads**: each `dot` runs at ~128 GB/s then
sleeps, and the histogram mean is over ticks in which the engine was *powered*. Read it as
"bandwidth while active" and divide by the `*_duty` column for a wall-clock average.

## What this does and does not license (M3 Pro)

**Does:** per-engine attribution on m3pro is trustworthy — the loaded engine moves and the others
stay at baseline in every case tested, including read/write split. CPU-driven bytes/s closes to
within ~6 %. The ANE bucket is calibrated and corroborated.

**Does not:**
1. **Above 32 GB/s per requestor (64 GB/s aggregate) the top bin is a catch-all** and the row is
   a **lower bound**. The CSV's `saturated` column flags it; never quote a flagged row as a
   bytes/s value.
2. **Below ~1 GB/s** (bin 0) a requestor is indistinguishable from idle, and a powered-but-idle
   engine shows a ~0.5 GB/s phantom floor. Baseline-subtract.
3. **Bursty engines over-report** unless divided by `*_duty` (see GPU above).
4. Write traffic is exercised (the ANE phase is ~50 % writes) but not independently
   byte-calibrated; write-allocate behaviour remains uncalibrated on both machines.

**Collection note.** CoreML does not engage the ANE for the first **~8 s** of a run — `ANE0`
reports no ticks at all while it compiles, even though throughput looks healthy. Warm up ≥10 s
before the measurement window, or the ANE will read as having moved nothing.

# AMC counter closure calibration — M2 Pro

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

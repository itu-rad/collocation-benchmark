# Brief: per-engine DRAM counters do not work on the M3 Pro

**Status:** blocking one exhibit in §5.2. Types cells are unaffected; the bandwidth
dose–response is.

## Goal

Read **per-engine DRAM byte counters** (CPU / GPU / **ANE** read+write) on
`itu-mac` (Apple M3 Pro, 18 GB, macOS 26.5.2) at ≥1 Hz **without root**, so a
collocation experiment can show *which engine moved the bytes*.

This underpins the section's flagship claim — "separation is not isolation":
when a background workload is moved off the GPU onto the ANE, compute contention
disappears but the foreground still degrades, because both engines share one
unified-memory pool. Showing the ANE background *still moving DRAM bytes* is the
finding. An aggregate bandwidth or energy number cannot show it.

## What works, and where

| | M2 Pro (`mac623807`, 16 GB) | M3 Pro (`mac624090` / itu-mac, 18 GB) |
|---|---|---|
| `AMC Stats / Perf Counters / * DCS RD\|WR` present | yes | **yes — 48 channels** |
| those channels return non-zero | **yes** | **no — all zero, idle and under load** |
| `Energy Model / AMCC` (mJ) | yes | **yes, responsive** |

The sampler is `scripts/amc_bandwidth_sampler.c` (+ `.py` wrapper, auto-compiles,
no root). It works correctly on the M2 Pro: CPU bucket 1.8 → 24.7 GB/s under an
induced ~19.5 GB/s stream; GPU bucket 0.8 → 70 GB/s under a matmul loop.

## The discrepancy

On the M3 Pro the channels **exist and read zero**. Verified by two independent
tools — the sampler and `scripts/ioreport_bw_probe.c`, which enumerates all 9056
IOReport channels — under sustained heavy memory traffic (a 600 MB
`bytearray` copy loop) as well as idle. No AMC DCS channel ever shows a non-zero
delta.

What *does* respond on the same machine, at the same moment:

| `Energy Model / AMCC` | mJ per 1 s sample |
|---|---|
| idle | 153, 151, 155 |
| heavy memory traffic | 1553, 1528, 1527 |

≈10× dynamic range, tight across repeats. So the memory controller is
instrumented and reporting — just not through the per-requestor byte channels.

## Already ruled out

Three things were fixed *before* this conclusion, so they are not the cause:

1. **Subscription failure.** `IOReportCopyChannelsInGroup(CFSTR("AMC Stats"))`
   returns a dictionary the subscription rejects on M3 (`IOReportCreateSubscription
   failed`), while the same channels are present in the all-channel copy. The
   sampler now retries unscoped.
2. **Name-shape difference.** M2 names a channel `DIE0 GFX DCS RD`; M3 names the
   same channel `GFX DCS RD`. The sampler used the `DIE` prefix to separate the
   memory-controller aggregate from the per-requestor breakdown, so on M3 every
   per-requestor channel was misread as the aggregate. It now keys on the
   requestor token itself.
3. **Silent zeros.** With both fixed, the sampler produced a clean run and an
   all-zero CSV — downstream indistinguishable from a machine that moved no
   memory. It now checks the first real sample and exits 3 with the reason, and
   the Python wrapper propagates that exit.

Also not the cause: permissions (no root needed on either machine, and the probe
reads other channels fine), and Xcode CLT (present; the sampler compiles).

## The question to chase

**Is there any unprivileged path to per-engine DRAM byte attribution on M3-family
Apple silicon?** Candidate directions, none verified:

- A different IOReport group/subgroup on M3 carrying per-requestor traffic under
  another name (the probe dumps all 9056 channels — grep for other `DCS`, `DRAM`,
  `BW`, or per-agent counters that *do* move under load).
- Whether the M3 AMC channels need an explicit enable/subscribe mode that the M2
  did not (e.g. a state/config channel that gates counting).
- `powermetrics` with a bandwidth sampler on this OS build (root; check whether
  the sampler is even advertised in `-h`).
- Third-party readers that track newer silicon: `socpowerbud`, `macpm`,
  `asitop` — do any report per-engine bandwidth on M3?

## Reproduce

```bash
# on itu-mac, repo root
clang -O2 -o /tmp/iorprobe scripts/ioreport_bw_probe.c -framework CoreFoundation
/tmp/iorprobe | grep "AMC Stats"          # channels are listed
./scripts/.build/amc_bandwidth_sampler -i 500 -n 2 -o /tmp/x.csv; echo $?   # exits 3

# generate memory traffic, then re-check that AMCC moves but AMC DCS does not
python3 -c "
import time
a=bytearray(600*1024*1024); t=time.time()
while time.time()-t<12: a[:300*1024*1024]=a[300*1024*1024:]" &
/tmp/iorprobe | grep -E "AMCC|AMC Stats.*DCS"
```

## Fallbacks if it cannot be made to work

1. Run the dose–response on the **M2 Pro** and name the machine — keeps
   per-engine bytes, costs a third machine in the paper.
2. Use **AMCC energy** on the M3 Pro — stays on the reported machine, but is a
   single aggregate: it can confirm a dose, not attribute it to an engine.

Relevant files: `scripts/amc_bandwidth_sampler.{c,py}`,
`scripts/ioreport_bw_probe.c`, `scripts/preflight_bandwidth_counters.sh`,
`evaluation/contention/AMC_CALIBRATION.md`, `evaluation/contention/contention.md`.

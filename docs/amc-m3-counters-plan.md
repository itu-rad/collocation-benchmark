# Plan: restore per-engine DRAM counters on the M3 Pro

**Status:** root cause found; replacement counter source **validated on
`itu-mac` for CPU, GPU and ANE**, and the **dual-backend sampler is implemented
and verified on both machines**, and the tooling and prose corrections are
landed (2026-09-05). **All six steps done.** Supersedes the diagnosis in
[`amc-m3-counters-brief.md`](amc-m3-counters-brief.md).

**Bottom line:** per-engine DRAM attribution *is* available on the M3 Pro,
unprivileged, at ≥1 Hz — including the ANE. It lives in a different IOReport
group than on the M2, in a different format. The §5.2 dose–response exhibit does
not need to move machines or fall back to aggregate energy.

## 1. What is actually wrong

The brief concluded "the channels exist and read zero". That is not what happens.

On the M3 Pro the `AMC Stats` channels **never reach the sample at all**:

| stage | M2 Pro | M3 Pro |
|---|--:|--:|
| `IOReportCopyAllChannels` → `AMC Stats` channels | 117 | 128 |
| survive into `subbedChannels` | 117 | 128 |
| **survive into `IOReportCreateSamples`** | **117** | **0** |

A per-group census of *every* group on the machine shows `AMC Stats`
(driver `AppleH15MemCacheController`) is the **only** group on the M3 dropped
between subscription and sampling. Every other group samples fine.

Group-scoped subscription fails outright, and it is not a size or ordering
problem — subscribing to a **single** AMC channel fails too:

| N channels subscribed | 1 | 2 | 8 | 32 | 128 |
|---|---|---|---|---|---|
| M2 Pro | ok | ok | ok | ok | ok |
| M3 Pro | FAILED | FAILED | FAILED | FAILED | FAILED |

So this is a flat per-driver refusal by `AppleH15MemCacheController`, not a
counter that reads zero. When the subscription is built from the all-channels
dict, `IOReportCreateSubscription` returns non-NULL (the other ~8900 channels
subscribe fine) and silently discards the AMC ones.

**Why the sampler misreported it.** The `n_live` guard added in fix (3) iterates
the *sample* looking for AMC `DCS RD/WR` channels and reports "present but ALL
read zero" when it finds none non-zero. On the M3 it finds *no such channels at
all* — zero matches, not zero values — and prints the "all read zero" message.
The brief's central claim ("the memory controller is instrumented and reporting,
just not through the per-requestor byte channels") followed from that artifact.
`ioreport_bw_probe.c` agreed only because it enumerates `chans` (where the
channels do exist) but reads deltas through the same dropped subscription.

Root is not a workaround: there is no passwordless sudo on `itu-mac` to test it,
and it is moot — the replacement below needs no privileges.

## 2. The replacement source

The M3 Pro publishes per-requestor DRAM bandwidth under the **PMP** group
(driver `RTBuddyIOReportingEndpoint`, 311 channels, subscribes and samples
normally):

```
PMP / DCS BW / EACC0 {RD,WR,RD+WR}     e-core cluster
PMP / DCS BW / PACC0 {RD,WR,RD+WR}     p-core cluster
PMP / DCS BW / AGX   {RD,WR,RD+WR}     GPU
PMP / DCS BW / ANE0  {RD,WR,RD+WR}     ANE   <-- the flagship claim
PMP / DCS BW / {ISP,DISP,DISPEXT0,...} other
PMP / DCS BW / RD+WR                   controller aggregate
```

`DCS` is the same DRAM-command-scheduler layer the M2's `AMC Stats … DCS RD/WR`
counters measure, so this is DRAM traffic, not fabric traffic. (`PMP / AF BW / *`
is the fabric-side equivalent and should **not** be used for byte claims.)

**These are not byte counters.** They are `kIOReportFormatState` (fmt=2)
channels used as 32-bin histograms: bin *i* is labelled with its upper edge in
GB/s, and its residency is the number of PMP ticks spent in that bandwidth bin
during the interval. Per-requestor channels use 1 GB/s bins (ceiling 32 GB/s);
the aggregate uses 2 GB/s bins (ceiling 64 GB/s). Both existing tools missed
these because they filter on group `AMC` and read values with
`IOReportSimpleGetIntegerValue`, which returns garbage for fmt=2.

Mean bandwidth over an interval is `Σ(residency_i × midpoint_i) / Σ(residency_i)`,
with `midpoint_i = label_i − width/2` and `width = label_i / (i+1)` (uniform bins,
so this adapts to the aggregate's 2 GB/s bins and to the M2's wider ones).

### Validated on `itu-mac` (2026-09-05)

All three engines checked against loads whose bytes are known or whose engine
placement is independently corroborated. **Both gates pass.**

**CPU** — known-byte read stream (read-only sum over a 2 GB buffer, the same
unambiguous-bytes method as `AMC_CALIBRATION.md`), smoothly paced:

| delivered (GB/s) | `PACC0 RD` reported | ratio |
|--:|--:|--:|
| 3.00 | 3.13 | 1.04 |
| 6.00 | 5.77 | 0.96 |
| 12.00 | 11.21 | 0.93 |
| 18.00 | 16.97 | 0.94 |
| 25.00 | 23.44 | 0.94 |
| 30.00 | 28.10 | 0.94 |

Slope ≈ 0.94, stable across the range. The ~6 % deficit is the expected direction
for a DRAM-side counter (reads served by the system-level cache never reach DCS).
An unthrottled stream (72.5 GB/s measured) pinned both `PACC0` and the aggregate
into their top bins — the counters agree with reality rather than saturating
spuriously.

**GPU** — paced `torch.dot` on `mps` (`scripts/../gpu_stream.py` method, i.e. the
GPU phase of `amc_calibration.py`):

| delivered (GB/s) | `AGX RD` reported | `PACC0 RD` |
|--:|--:|--:|
| 5.00 | 11.07 | 0.58 |
| 11.99 | 17.12 | 0.59 |
| 19.99 | 21.03 | 0.61 |
| 128.87 | 30.70 (pegged) | 0.56 |

Attribution is exact — `PACC0` never leaves its 0.56–0.61 idle baseline while
`AGX` tracks — but the **absolute value over-reports for bursty loads**. Each
`dot` runs at ~128 GB/s and then sleeps; the histogram mean is over ticks in
which the GPU was *powered*, so it reports "bandwidth while active", not the
wall-clock time average. The smooth CPU load did not expose this because it was
genuinely continuous. See limit 4.

**ANE** — a deliberately ANE-friendly fp16 conv stack
(`scripts/make_ane_model.py`, 24 convs, 16.8 MB activations/layer), driven by
`scripts/ane_load.py`:

| | `ANE0 RD+WR` | `AGX RD+WR` | corroboration |
|---|--:|--:|---|
| `compute_units=CPU_ONLY` (6.1 pred/s) | *no ticks* | 0.50 (idle) | ANE `IOP State` = Off 100 % |
| `compute_units=CPU_AND_NE` (31.2 pred/s) | **19.75** | 0.50 (idle) | `IOP State` = **Running 100 %**, `Energy Model / ANE` = **6591 mJ** vs 0 idle |

**The flagship claim is measurable.** ANE DRAM traffic sits ~20× above the
~1 GB/s floor, with the GPU provably idle, and two independent channels
(`ANE / IOP State`, `Energy Model / ANE`) confirm the placement. The read/write
split is available too (9.02 RD / 8.92 WR).

> **Operational finding — CoreML ANE warm-up.** The ANE does not engage for the
> **first ~8 s** of a run: CoreML serves early predictions on the CPU while the
> model is compiled for the ANE, and `ANE0` reports no ticks at all during that
> window even though throughput looks healthy. It then steps to a steady
> ~19.7 GB/s. Any ANE background must be warmed up before the measurement window
> opens, or the exhibit will read "the ANE moved nothing". This compounds the
> existing first-repetition warm-up rule.
>
> This is also why the ANE looked dead at first: the repo's exported
> `tmp/clip_vit_l14_vision.mlpackage` never reaches the ANE at all
> (`IOP State` = Off throughout, `CPU_AND_NE` only 2× faster than `CPU_ONLY`
> via a different CPU path). If §5.2 intends CLIP as the ANE background, that
> export needs fixing first — it is currently a CPU workload.

### Limits that must be documented

1. **Ceiling.** Per-requestor channels saturate above **32 GB/s** (top bin is a
   catch-all); the aggregate saturates above **64 GB/s**. The M3 Pro's bus is
   ~150 GB/s, so a saturating foreground *will* peg its bucket. `AF BW` has the
   same 32 GB/s ceiling, so there is no higher-range alternative. Acceptable for
   §5.2 — the ANE sits at ~20 GB/s, under the ceiling — but any *foreground* GPU
   bytes/s number on the M3 must be reported as a lower bound.
2. **Floor.** Bin 0 is a `<1 GB/s` catch-all, so a requestor below ~1 GB/s is
   indistinguishable from idle. Confirmed not to bite for the ANE (~20 GB/s).
3. **Power-gated blocks contribute no ticks**, so total ticks differ per channel
   and the mean is over *powered* time. `ANE0` reports nothing at all when the
   ANE is gated — which is the correct reading of "the ANE moved nothing", but is
   indistinguishable from the warm-up window above.
4. **Bursty loads over-report.** Because the mean is over powered ticks, a duty-
   cycled engine reports its active-phase bandwidth, not the wall-clock average
   (GPU at 5.00 GB/s delivered reads 11.07). Report the metric as **mean DRAM
   bandwidth while the engine is active**, and publish the active-tick fraction
   next to it; do not present it as a time-averaged bytes/s for bursty engines
   without dividing through by that fraction.

## 3. Implementation plan

**Step 1 — make the sampler backend-aware. DONE.** In `scripts/amc_bandwidth_sampler.c`,
select at startup:

- *amc backend* (M2): today's `AMC Stats … DCS RD/WR` byte counters. Unchanged.
- *pmp backend* (M3): `PMP / DCS BW / *` histograms, via `IOReportStateGetCount`
  / `IOReportStateGetNameForIndex` / `IOReportStateGetResidency`.

Pick the backend by *what actually samples*, not by chip name: take one sample,
and if no `AMC Stats … DCS` channel appears in it, fall back to PMP. Fail with
exit 3 only when **neither** backend yields a live channel.

**Step 2 — keep the CSV schema byte-valued. DONE.** Emit
`bytes = mean_GBps × 1e9 × dt_s` per bucket so
`timestamp,dt_s,{cpu,gpu,ane,other}_{rd,wr},total_rd,total_wr,total_gbps` is
unchanged. This keeps `staged_lib.py`, `analyze_staged.py`, `calibrate_bytes.py`,
`amc_calibration.py`, `collect_e5.sh` and the radt listener patch
(`evaluation/radt-patches/0001-amc-bandwidth-listener.patch`) working untouched.
Requestor→bucket map for PMP: `PACC*`/`EACC*` → cpu, `AGX` → gpu, `ANE*` → ane,
rest → other; `PMP / DCS BW / RD+WR` is the aggregate/total.
Add a `backend` column or a header comment so a CSV is self-describing, and
record saturation: if any bucket sits in its top bin for a sample, flag it, so a
pegged reading is never silently reported as a bytes/s value.

**Step 3 — validate GPU and ANE on the M3. DONE (2026-09-05), both pass.**
Results in §2 above. Two follow-ups fall out of it, neither blocking the counter
work: (a) fix or replace the CLIP CoreML export, which never reaches the ANE;
(b) add an ANE warm-up (≥10 s) to any collection that uses an ANE background.

**Step 4 — extend the fixed tooling. DONE.**
- `ioreport_bw_probe.c`: read fmt=2 channels via the state accessors and stop
  reporting them through `IOReportSimpleGetIntegerValue`; report the
  copied-vs-subscribed-vs-sampled counts per group, which is what would have
  caught this immediately.
- `preflight_bandwidth_counters.sh`: accept the PMP path as a pass, and check
  channels *survive into a sample* rather than merely existing.

**Step 5 — correct the record. DONE.** `evaluation/contention/contention.md` §"The M3
Pro does not populate the per-engine DRAM counters (2026-09-03)" and its
capability table row are now wrong; commit `7011f9f` asserts the same. Rewrite
both once step 3 passes. Add an M3 section to
`evaluation/contention/AMC_CALIBRATION.md` with the table above plus the GPU/ANE
results, and state the 32 GB/s ceiling and 1 GB/s floor alongside the numbers.

**Step 6 — drop the fallbacks. DONE.** The brief's fallback 1 (move the dose–response
to the M2, costing a third machine in the paper) and fallback 2 (AMCC energy, no
attribution) are both unnecessary: step 3 passed, so §5.2 keeps per-engine bytes
on the machine the rest of the paper reports.

### Steps 1-2 as built (2026-09-05)

`scripts/amc_bandwidth_sampler.c` now selects its backend from what actually
samples: it tries the cheap AMC-group-scoped subscription first (the M2 path,
~117 channels instead of ~9000), and falls back to an all-channel subscription
in which it looks for live AMC byte counters and then for live PMP histograms.
Exit 3 now fires only when **neither** backend yields a live channel.

Six numeric columns were appended: `{cpu,gpu,ane,other}_duty`, `backend`
(0=amc, 1=pmp) and `saturated`. They are numeric on purpose —
`staged_lib.load_bandwidth_csv` and `summarize` both `float()` every field and
silently drop rows that fail, so a text column would have blanked the CSV.

Verified end to end:

| | backend | load | reported |
|---|---|---|--:|
| M2 Pro | `amc` (42 live) | 15.00 GB/s CPU stream | `cpu` 16.9 GB/s (incl. ~1.8 baseline) |
| M3 Pro | `pmp` (37 live) | 20.00 GB/s CPU stream | `cpu` 19.4 GB/s (ratio 0.97) |
| M3 Pro | `pmp` | ANE conv model, warmed up | **`ane` 19.94 GB/s**, `gpu` 0.07, `ane_duty` 1.00, `saturated` 0 |

`summarize()` and `bandwidth_window_stats()` parse 3/3 rows on both backends
with no changes to either.

### Steps 4-5 as built (2026-09-05)

`ioreport_bw_probe.c` now prints a per-group **COPIED -> SUBSCRIBED -> SAMPLED**
census and marks any group that enumerates but never samples as `DROPPED`, and
reads each channel with the accessor its format requires (fmt=2 histograms via
the state accessors rather than `IOReportSimpleGetIntegerValue`). On m3pro the
one-line diagnosis now falls straight out of it:

```
COPIED  SUBBED  SAMPLED  group  <<driver>>
128     128     0        AMC Stats  <<AppleH15MemCacheController>>   <== DROPPED
311     311     311      PMP  <<RTBuddyIOReportingEndpoint>>
```

`preflight_bandwidth_counters.sh` checks survival into a sample rather than mere
existence, and adds a Probe 5 that builds and runs the sampler itself and reports
which backend it picked — the authoritative check, since that is what collection
records. Verdicts: M2 Pro "IOReport AMC byte counters -> COUNTER-BACKED, exact";
m3pro "IOReport PMP DCS BW histograms -> COUNTER-BACKED (derived bytes, 32 GB/s
ceiling)".

Prose corrected in `contention.md` (the 2026-09-03 section and the device-
asymmetry table row; the M2 Pro's validation numbers were also being attributed
to m3pro), `AMC_CALIBRATION.md` (new M3 section with all three calibration
tables), and the stale machine notes in `calibrate_bytes.py` and
`generate_stage_configs.py`. Worth noting from the latter: the dose ladder
(4/8/12/16 GB/s) and the matched level (12 GB/s) all sit under the 32 GB/s
per-requestor ceiling, so no cell is measured by a saturated counter — and the
measured ANE ceiling of 19.9 GB/s independently confirms the ~20 GB/s bound that
file already assumed.

## Reproduce

```bash
# on itu-mac, repo root — scratch tools used for this investigation
clang -O2 -o /tmp/ior_stage    scripts/ior_stage.c    -framework CoreFoundation
clang -O2 -o /tmp/pmp_bw_probe scripts/pmp_bw_probe.c -framework CoreFoundation
/tmp/ior_stage        # AMC Stats: 128 copied, 128 subscribed, 0 sampled
/tmp/pmp_bw_probe 3 1.0 "DCS BW"   # per-requestor GB/s, responds to load

# ANE check (conda env benchmark_macos); allow >10 s of warm-up before sampling
python scripts/make_ane_model.py 128 256 24 /tmp/ane_conv.mlpackage
python scripts/ane_load.py /tmp/ane_conv.mlpackage 30 CPU_AND_NE &
sleep 12 && /tmp/pmp_bw_probe 3 1.0 "DCS BW"
```

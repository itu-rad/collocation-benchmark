#!/usr/bin/env python3
"""AMC counter closure calibration (Apple silicon).

Drives a KNOWN DRAM read bandwidth (numpy over buffers >> LLC, read-only so the
byte count is unambiguous — no write-allocate guesswork) and compares it to what
the Apple Memory Controller per-requestor counters report. Derives:

  * the calibration factor (known GB/s / AMC-reported delta),
  * which bucket the CPU traffic lands in (agent->bucket map),
  * the residual/unattributed fraction,

so the staged-contention bytes/s axis can be corrected and the 'impossible
totals' (counter GB/s exceeding the LPDDR5 spec) explained. Run on an idle M2.
"""
import os
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "scripts"))
import numpy as np  # noqa: E402
from amc_bandwidth_sampler import AMCBandwidthSampler, summarize  # noqa: E402

N = 96_000_000               # 96M float64 = 768 MB/array; a,b = 1.5 GB >> LLC
BYTES_PER_ITER = 2 * N * 8   # np.dot reads a and b fully, writes nothing


def _sample(label, secs, work):
    tmp = tempfile.mktemp(suffix=f"_{label}.csv")
    s = AMCBandwidthSampler(label=label, out=tmp, interval=0.5).start()
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < secs:
        work()
        iters += 1
    elapsed = time.perf_counter() - t0
    s.stop()
    return summarize(tmp), iters, elapsed


def main():
    print(f"allocating 2x {N*8/1e9:.2f} GB buffers ...", flush=True)
    a = np.random.rand(N)
    b = np.random.rand(N)

    def read_load():
        # 2 full-array DRAM reads, no writes -> exactly BYTES_PER_ITER moved
        float(np.dot(a, b))

    def idle():
        time.sleep(0.05)

    print("=== idle baseline (6 s) ===", flush=True)
    base, _, _ = _sample("calib_idle", 6, idle)
    print("  ", {k: round(v, 2) for k, v in base.items()})

    print("=== known CPU read load (20 s) ===", flush=True)
    load, iters, elapsed = _sample("calib_load", 20, read_load)
    known = BYTES_PER_ITER * iters / elapsed / 1e9
    print("  ", {k: round(v, 2) for k, v in load.items()})

    d = {e: load[f"mean_{e}_gbps"] - base[f"mean_{e}_gbps"]
         for e in ("cpu", "gpu", "ane", "other")}
    d_total = load["mean_total_gbps"] - base["mean_total_gbps"]

    print("\n================ CALIBRATION =================")
    print(f"known delivered CPU read bandwidth : {known:8.1f} GB/s "
          f"({iters} iters / {elapsed:.1f} s)")
    print(f"AMC delta by bucket (load - idle)  : "
          f"cpu={d['cpu']:.1f}  gpu={d['gpu']:.1f}  ane={d['ane']:.1f}  "
          f"other={d['other']:.1f}  GB/s")
    print(f"AMC delta total                    : {d_total:8.1f} GB/s")
    print(f"load max total                     : {load['max_total_gbps']:8.1f} "
          f"GB/s   (LPDDR5 M2 Pro spec ~200 GB/s)")
    if d["cpu"]:
        print(f"factor  known / AMC-cpu-delta      : {known / d['cpu']:.3f}")
    if d_total:
        print(f"factor  known / AMC-total-delta    : {known / d_total:.3f}")
    # where did the known traffic land?
    if d_total:
        frac = {e: d[e] / d_total for e in d}
        print(f"attribution of the delta           : "
              + "  ".join(f"{e}={frac[e]*100:.0f}%" for e in d))
    print("=============================================")


if __name__ == "__main__":
    main()

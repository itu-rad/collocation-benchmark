"""Ecologically-valid 3D-UNet serving analysis: what MLPerf's order-/load-independent
numbers hide on the workload's REAL deployment (single on-prem device, batch cohort
reprocessing + modest/bursty arrivals).

Two results, both trace-driven from the measured per-study service times
(evaluation/unet3d/results_mps_r1.csv, 42 KiTS19 studies, 18x inference spread):

  A. BATCH COHORT REPROCESSING (M#2): a nightly batch of N studies on one GPU. MLPerf
     Offline reports THROUGHPUT = N / makespan, which is ORDER-INSENSITIVE — identical for
     any schedule. But mean time-to-result (flow time) and small-study latency depend
     heavily on order. SJF (shortest-first, using the pre-inference-known subvolume count)
     vs FIFO: the win MLPerf's metric structurally cannot express.

  B. HEAD-OF-LINE BLOCKING (H5) at MODEST/bursty load (NOT saturation): a routine 10s study
     that arrives just behind a 185s trauma study waits ~185s. MLPerf SingleStream (closed-
     loop, no queue) reports its latency as ~its service time. Real modest-load p99 is
     inflated ~order of magnitude — and only for the small studies.

Model: the GPU is the single shared server; service time per study = measured inference_s
(preprocessing overlaps on CPU except pipeline fill). Study "size" = subvolume count, known
after preprocessing and BEFORE inference (so a size-aware scheduler is realizable).
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import numpy as np

CSV = Path("evaluation/unet3d/results_mps_r1.csv")
SEED = 1234


def load():
    rows = [r for r in csv.DictReader(open(CSV)) if not r.get("error")]
    svc = np.array([float(r["inference_s"]) for r in rows])
    size = np.array([int(r["n_subvolumes"]) for r in rows])
    return svc, size


# ---------------------------------------------------------------------------
# A. Batch cohort reprocessing: FIFO (random arrival order) vs SJF
# ---------------------------------------------------------------------------

def batch_flow_times(svc, order):
    """Single server, all jobs present at t=0, processed in `order`.
    Returns completion time per job (= flow time, since arrival=0)."""
    comp = np.cumsum(svc[order])
    # map back to original index -> completion
    out = np.empty_like(comp)
    out[order] = comp
    return out


def analyze_batch(svc, size, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n = len(svc)
    makespan = svc.sum()
    mlperf_throughput = n / makespan  # order-insensitive

    small = size <= np.quantile(size, 0.25)  # the routine/small studies

    fifo_mean, fifo_small = [], []
    for _ in range(n_boot):
        order = rng.permutation(n)                     # random arrival order = FIFO
        ft = batch_flow_times(svc, order)
        fifo_mean.append(ft.mean())
        fifo_small.append(ft[small].mean())
    # SJF: shortest predicted size first (deterministic)
    sjf_order = np.argsort(size, kind="stable")
    sjf_ft = batch_flow_times(svc, sjf_order)

    return {
        "n": n, "makespan_s": makespan,
        "mlperf_throughput_reqhr": mlperf_throughput * 3600,
        "fifo_mean_flow_s": st.mean(fifo_mean),
        "sjf_mean_flow_s": sjf_ft.mean(),
        "fifo_small_flow_s": st.mean(fifo_small),
        "sjf_small_flow_s": sjf_ft[small].mean(),
        "flow_improvement_x": st.mean(fifo_mean) / sjf_ft.mean(),
        "small_improvement_x": st.mean(fifo_small) / sjf_ft[small].mean(),
        "n_small": int(small.sum()),
    }


# ---------------------------------------------------------------------------
# B. Head-of-line blocking under MODEST open-loop load (single-server FIFO M/G/1)
# ---------------------------------------------------------------------------

def simulate_openloop(svc, size, rho, n_req=20000, sjf=False, seed=SEED):
    """Poisson arrivals at utilization rho; service sampled from measured svc.
    Returns per-request (latency, size). Non-preemptive; SJF picks shortest waiting."""
    rng = np.random.default_rng(seed)
    mean_svc = svc.mean()
    lam = rho / mean_svc
    # sample services + sizes jointly (index into the empirical joint distribution)
    idx = rng.integers(0, len(svc), n_req)
    s = svc[idx]
    sz = size[idx]
    inter = rng.exponential(1.0 / lam, n_req)
    arr = np.cumsum(inter)

    lat = np.empty(n_req)
    if not sjf:
        # FIFO: closed-form recursion
        comp_prev = 0.0
        for i in range(n_req):
            start = max(arr[i], comp_prev)
            comp = start + s[i]
            lat[i] = comp - arr[i]
            comp_prev = comp
    else:
        # non-preemptive SJF via event loop over the waiting set
        import heapq
        server_free = 0.0
        # process in arrival order into a waiting pool, serve shortest available
        # event-driven: advance server; among arrived-and-unserved pick min size
        order_by_arr = np.argsort(arr)
        pending = []  # heap of (size, arrival, idx)
        ai = 0
        done = 0
        t = 0.0
        completed = np.zeros(n_req, bool)
        while done < n_req:
            # admit all arrivals up to server_free time
            while ai < n_req and arr[order_by_arr[ai]] <= max(server_free, t):
                j = order_by_arr[ai]
                heapq.heappush(pending, (sz[j], arr[j], j))
                ai += 1
            if not pending:
                # jump to next arrival
                if ai < n_req:
                    t = arr[order_by_arr[ai]]
                    server_free = max(server_free, t)
                    continue
                else:
                    break
            _, aj, j = heapq.heappop(pending)
            start = max(server_free, arr[j])
            comp = start + s[j]
            lat[j] = comp - arr[j]
            server_free = comp
            done += 1
    warm = n_req // 5  # drop warmup
    return lat[warm:], sz[warm:]


def analyze_hol(svc, size):
    small_thr = np.quantile(size, 0.25)
    small_svc_mean = svc[size <= small_thr].mean()
    out = {"small_svc_mean_s": small_svc_mean, "rho": {}}
    for rho in (0.2, 0.35, 0.5):
        lat, sz = simulate_openloop(svc, size, rho, sjf=False)
        latj, szj = simulate_openloop(svc, size, rho, sjf=True)
        sm = sz <= small_thr
        smj = szj <= small_thr
        out["rho"][rho] = {
            "small_p50": float(np.percentile(lat[sm], 50)),
            "small_p95": float(np.percentile(lat[sm], 95)),
            "small_p99": float(np.percentile(lat[sm], 99)),
            "small_p99_inflation_x": float(np.percentile(lat[sm], 99) / small_svc_mean),
            "sjf_small_p99": float(np.percentile(latj[smj], 99)),
            "all_p99": float(np.percentile(lat, 99)),
        }
    return out


def main():
    svc, size = load()
    print(f"=== service-time distribution (n={len(svc)}) ===")
    print(f"inference_s: mean {svc.mean():.1f}  median {np.median(svc):.1f}  "
          f"min {svc.min():.1f}  max {svc.max():.1f}  CV {svc.std()/svc.mean():.2f}")
    print(f"MLPerf SingleStream p90 (service) = {np.percentile(svc,90):.1f}s\n")

    print("=== A. BATCH COHORT REPROCESSING (nightly, single GPU) — FIFO vs SJF ===")
    a = analyze_batch(svc, size)
    print(f"cohort N={a['n']} studies, makespan {a['makespan_s']/60:.1f} min")
    print(f"MLPerf Offline throughput = {a['mlperf_throughput_reqhr']:.1f} studies/hr "
          f"— IDENTICAL for FIFO and SJF (order-insensitive)")
    print(f"mean time-to-result:  FIFO {a['fifo_mean_flow_s']/60:.1f} min  |  "
          f"SJF {a['sjf_mean_flow_s']/60:.1f} min  ->  {a['flow_improvement_x']:.2f}x better")
    print(f"SMALL/routine studies (n={a['n_small']}) mean time-to-result: "
          f"FIFO {a['fifo_small_flow_s']/60:.1f} min  |  SJF {a['sjf_small_flow_s']/60:.1f} min "
          f"->  {a['small_improvement_x']:.2f}x better")
    print(f"  => MLPerf's throughput number is the SAME for both schedules; it cannot see "
          f"the {a['small_improvement_x']:.1f}x time-to-result win.\n")

    print("=== B. HEAD-OF-LINE BLOCKING at MODEST/bursty load (not saturation) ===")
    h = analyze_hol(svc, size)
    print(f"small/routine study mean service time = {h['small_svc_mean_s']:.1f}s "
          f"(what MLPerf SingleStream reports as its latency)")
    for rho, d in h["rho"].items():
        print(f"  load rho={rho}: small-study p50 {d['small_p50']:.0f}s  p95 {d['small_p95']:.0f}s  "
              f"p99 {d['small_p99']:.0f}s  ({d['small_p99_inflation_x']:.1f}x its service) "
              f"| SJF p99 {d['sjf_small_p99']:.0f}s")
    print(f"  => at realistic modest load, a routine study's TAIL latency is inflated ~order "
          f"of magnitude by large studies ahead of it — invisible to MLPerf's closed-loop "
          f"SingleStream; size-aware scheduling (SJF) recovers it.")


if __name__ == "__main__":
    main()

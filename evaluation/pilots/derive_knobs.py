#!/usr/bin/env python3
"""Rules engine: pilot measurements + pre-registered rules -> knobs.yml.

Every experiment knob derives from a stated rule (ids below) whose inputs come
from the serial pilots (run_pilots.py) or from committed data (E1/E2 warm-up
horizons). knobs.yml is the single provenance artifact: apply_knobs.py patches
the committed configs from it, generate_knob_tables.py renders the paper
tables from it, verify_knobs.py fills its `verified:` fields post-collection.

Rule registry (pre-registered; the paper's knob tables cite these ids):
  R-LAMBDA-BELOW-SAT  λ = ρ / median serial service time of the SLOWEST arm in
                      the comparison, ρ = 0.6 (legal 0.5–0.8, choice stated);
                      identical for every arm on a device.
  R-LAMBDA-NFLIGHT    contended cells run closed-loop with N=2 in flight (the
                      smallest concurrency that forces stage overlap); no λ.
  R-LAMBDA-SAT        saturating (all enqueued at once); throughput-only —
                      per-query latency is never reported from these cells.
  R-QDEPTH            queue_depth = max_queries for open-loop cells (cannot
                      block by construction); >= total samples for saturating.
                      Verified from the arrivals sidecar (0 blocked puts).
  R-WARMUP            k = 2 x rolling-median flatness point on the pilot
                      (detect_warmup); ANE first-call outliers excluded AND
                      reported separately, never folded into k.
  R-NTIMING           timing cells: 110 queries x R (>=500 pooled post-warmup
                      for the p95 gate; refined from 40 on 2026-07-14).
  R-NQUALITY          quality: 120 questions per arm, dedicated serial runs
                      (Wilson 95% half-width <= ~9 points at p~0.5).
  R-REPS              R = 10 for cheap cells (E1/E2/E3), 5 for expensive
                      (E4/E6): runs are the replication unit.
  R-TIMEOUT           timeout(s) = ceil(5 x expected duration) where expected
                      = max_queries/λ + 3 x median service (open-loop) or
                      max_queries x median (serial). SECONDS everywhere —
                      normalizes the previous ms/s incoherence.
  R-E6-HEADROOM       foreground λ = 0.4 x capacity (B=0 sits at ~40%
                      utilization), held fixed across all B.
  R-INTENSITY         E3' co-runner intensities = {25,50,75,100}% of pilot
                      R_max (saturated rate).
  R-PRECEDENT         workload-defining params from external precedent, held
                      constant across arms: top_k=3 (+top_k=10 sensitivity on
                      multi-hop core arms), max_retries=2, max_tokens 256/128.

Usage:
    python evaluation/pilots/derive_knobs.py [--variant e3=vqa|dose_response]
        [--variant e6=torchvision|rag_indexing] [--e3-p95 raise|drop]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import glob as globmod
import math
import re
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_lib as pl  # noqa: E402

REPO_ROOT = HERE.parent.parent
RESULTS_DIR = HERE / "results"
KNOBS_PATH = HERE / "knobs.yml"

RHO = 0.6            # R-LAMBDA-BELOW-SAT load factor (stated choice in 0.5-0.8)
E6_RHO = 0.4         # R-E6-HEADROOM
N_TIMING = 110       # R-NTIMING (refined 2026-07-14: (N-1)*R >= 500 pooled
                     # post-warmup samples at R=5 meets the p95 gate; was 40,
                     # which pooled only 195 — rule collision found by
                     # validate_pass.py)
N_QUALITY = 120      # R-NQUALITY
DEVICE_NAME = {"mlx": "m2pro", "cuda": "gb10"}


# ---------------------------------------------------------------------------
# Pilot ingestion
# ---------------------------------------------------------------------------

def summarize_pilot(cell_id: str, device: str) -> dict | None:
    """Aggregate all runs of one pilot cell into service + warm-up summaries."""
    paths = sorted(RESULTS_DIR.glob(f"pilot_{cell_id}_{device}_r*.csv"))
    if not paths:
        return None
    all_lat, per_run_medians = [], []
    for p in paths:
        lat = pl.per_query_latencies(p)
        if lat:
            all_lat.append(lat)
    if not all_lat:
        return None
    # Warm-up on the first run's raw series; service stats pooled post-warm-up.
    # Non-converged detection on a short/retry-noisy pilot falls back to the
    # pre-registered LLM default k=2 (first-call kernel compile) — a spurious
    # large k would silently discard most of the pilot.
    wu = pl.detect_warmup(all_lat[0], window=5, epsilon=0.10)
    if not wu.converged and wu.k_fixed > 2:
        wu.k_fixed = 2
        wu.note = (wu.note + "; " if wu.note else "") + \
            "non-converged -> fallback k=2 (pre-registered LLM default)"
    pooled = []
    for series in all_lat:
        stats = pl.service_stats(series, wu.k_fixed, wu.outlier_idxs)
        if stats.get("n"):
            per_run_medians.append(round(stats["median"], 3))
        drop = set(range(wu.k_fixed)) | set(wu.outlier_idxs)
        pooled.extend(v for i, v in enumerate(series) if i not in drop)
    if not pooled:
        return None
    import statistics
    return {
        "cell": cell_id,
        "device": DEVICE_NAME[device],
        "csvs": [p.name for p in paths],
        "n_pooled": len(pooled),
        "warmup": {"k_star": wu.k_star, "k_fixed": wu.k_fixed,
                   "window": wu.window, "epsilon": wu.epsilon,
                   "converged": wu.converged, "note": wu.note,
                   "outlier_idxs": wu.outlier_idxs},
        "service_s": {
            "median": round(statistics.median(pooled), 4),
            "mean": round(sum(pooled) / len(pooled), 4),
            "min": round(min(pooled), 4), "max": round(max(pooled), 4),
        },
        "per_run_medians_s": per_run_medians,
    }


def committed_warmup(pattern: str, window: int, epsilon: float,
                     series_fn) -> dict | None:
    """Warm-up detection over already-committed CSVs (E1/E2 — zero-cost pilots)."""
    paths = sorted(globmod.glob(str(REPO_ROOT / pattern)))
    if not paths:
        return None
    results = []
    for p in paths:
        x = series_fn(p)
        if len(x) >= 2 * window:
            wu = pl.detect_warmup(x, window=window, epsilon=epsilon)
            results.append(wu)
    if not results:
        return None
    conv = [r for r in results if r.converged]
    use = conv or results
    k_star = max(r.k_star for r in use)
    return {"csvs_scanned": len(paths), "converged_runs": len(conv),
            "k_star_max": k_star, "k_fixed": max(1, 2 * k_star),
            "window": window, "epsilon": epsilon}


def _e2_step_series(path: str) -> list[float]:
    sys.path.insert(0, str(REPO_ROOT / "evaluation" / "overheads" / "modularity_overhead"))
    import modularity_lib as ml
    return [v / 1e9 for v in ml.parse_choreo_train_steps(path)]


# ---------------------------------------------------------------------------
# Knob assembly
# ---------------------------------------------------------------------------

def K(knob, value, rule, scope, inputs=None, applies_to=None, verification=None,
      note=None):
    e = {"knob": knob, "value": value, "rule": rule, "scope": scope,
         "verified": None}
    if inputs:
        e["inputs"] = inputs
    if applies_to:
        e["applies_to"] = applies_to
    if verification:
        e["verification"] = verification
    if note:
        e["note"] = note
    return e


def _lam(median_s: float, rho: float = RHO) -> float:
    return round(rho / median_s, 4)


def _timeout_open(n: int, lam: float, med: float) -> int:
    return math.ceil(5 * (n / lam + 3 * med))


def _timeout_serial(n: int, med: float) -> int:
    return math.ceil(5 * n * med)


def derive_e4(pilots: dict, device: str) -> list[dict]:
    """λ binds per (task, device): the monolith-vs-decomposed contrast is read
    WITHIN a difficulty, so 'slowest arm in the comparison' means slowest arm
    of that task — one λ across all arms of the task on the device."""
    tasks = {
        "factoid": ["e4_factoid_mono9b", "e4_factoid_decomp",
                    "e4_factoid_mono4b", "e4_factoid_shared"],
        "multihop": ["e4_multihop_mono9b", "e4_multihop_decomp"],
    }
    suffix = "mlx" if device == "mlx" else "cuda"
    entries = []
    any_pilot = False
    for task, arms in tasks.items():
        have = {a: pilots[(a, device)] for a in arms if (a, device) in pilots}
        glob_pat = f"evaluation/self_rag/configs/{task}_*_{suffix}.yml"
        if not have:
            entries.append(K(f"{task}.loadgen.config.rate", None,
                             "R-LAMBDA-BELOW-SAT", "config",
                             note="pending pilot on this device"))
            continue
        any_pilot = True
        slowest_id = max(have, key=lambda a: have[a]["service_s"]["median"])
        med = have[slowest_id]["service_s"]["median"]
        lam = _lam(med)
        entries += [
            K("loadgen.config.rate", lam, "R-LAMBDA-BELOW-SAT", "config",
              inputs={"pilot": slowest_id, "median_service_s": med, "rho": RHO,
                      "task": task,
                      "all_arm_medians_s": {a: have[a]["service_s"]["median"]
                                            for a in have}},
              applies_to=[glob_pat],
              verification="arrivals sidecar: 0 blocked puts; realized rate within 5% of λ"),
            K("loadgen.max_queries", N_TIMING, "R-NTIMING", "config",
              applies_to=[glob_pat]),
            K("loadgen.queue_depth", N_TIMING, "R-QDEPTH", "config",
              applies_to=[glob_pat],
              verification="arrivals sidecar: max block_s < 5 ms"),
            K("loadgen.timeout", _timeout_open(N_TIMING, lam, med), "R-TIMEOUT",
              "config", inputs={"unit": "seconds", "task": task},
              applies_to=[glob_pat]),
        ]
    if any_pilot:
        wu = pilots.get(("e4_factoid_mono9b", device))
        if wu:
            entries.append(K("warmup_k", wu["warmup"]["k_fixed"], "R-WARMUP",
                             "analysis", inputs={"pilot": wu["cell"],
                                                 **wu["warmup"]}))
    entries += [
        K("R", 5, "R-REPS", "driver"),
        K("n_quality", N_QUALITY, "R-NQUALITY", "driver",
          verification="Wilson 95% half-width <= 9 points at the observed rate"),
        K("top_k", {"factoid": 3, "multihop": 5}, "R-PRECEDENT", "config",
          inputs={"sensitivity": {"top_k": 10, "on": "multihop core arms"}},
          note="per task difficulty (retrieval differs across difficulties by "
               "design; contrasts are read within a difficulty)"),
        K("max_retries", 2, "R-PRECEDENT", "config"),
    ]
    return entries


def derive_e3(pilots: dict, device: str, variant: str, p95_policy: str) -> list[dict]:
    # vqa variant was M2-only; the dose_response (staged experiment) runs on
    # BOTH DUTs — co-runner ladders derive per device (no ANE on gb10).
    if variant == "vqa" and device != "mlx":
        return []
    entries = []
    if variant == "vqa":
        a = pilots.get(("e3_vqa_a", device))
        b = pilots.get(("e3_vqa_b", device))
        n_q = 100 if p95_policy == "raise" else N_TIMING
        entries.append(K("p95_policy", p95_policy, "R-NTIMING", "analysis",
                         note="raise: 100 q/run x R=10 -> 1000 pooled clears the >=500 gate"))
        if a and b:
            med = max(a["service_s"]["median"], b["service_s"]["median"])
            entries += [
                K("contended_mode", {"scheduler": "n_inflight_closed_loop",
                                     "n_inflight": 2},
                  "R-LAMBDA-NFLIGHT", "driver",
                  inputs={"slowest_mapping_median_s": med},
                  note="BLOCKED on the N-in-flight scheduler build (PAPER_TODO §2.2)"),
                K("loadgen.max_queries", n_q, "R-NTIMING", "config",
                  applies_to=["pipeline_configs/multimodal_vqa_mapping_*.yml"]),
                K("loadgen.queue_depth", n_q, "R-QDEPTH", "config",
                  applies_to=["pipeline_configs/multimodal_vqa_mapping_*.yml"]),
                K("loadgen.timeout", _timeout_serial(n_q, med), "R-TIMEOUT",
                  "config", inputs={"unit": "seconds"},
                  applies_to=["pipeline_configs/multimodal_vqa_mapping_*.yml"]),
                K("warmup_k", a["warmup"]["k_fixed"], "R-WARMUP", "analysis",
                  inputs={"pilot": "e3_vqa_a", **a["warmup"]}),
                K("ane_first_call", b["warmup"]["outlier_idxs"], "R-WARMUP",
                  "analysis", inputs={"pilot": "e3_vqa_b"},
                  note="excluded AND reported separately"),
            ]
        else:
            entries.append(K("loadgen.max_queries", None, "R-NTIMING", "config",
                             note="pending pilot"))
    else:  # dose_response (E3')
        corunners = [("e3p_c1_rmax", "c1_gpu"), ("e3p_c3_rmax", "c3")]
        if device == "mlx":  # no Neural Engine on GB10
            corunners.insert(1, ("e3p_c2_rmax", "c2_ane"))
        for cell, label in corunners:
            p = pilots.get((cell, device))
            if p:
                rmax = round(1.0 / p["service_s"]["median"], 3)
                entries.append(K(f"corunner_{label}_levels",
                                 [round(f * rmax, 3) for f in (0.25, 0.5, 0.75, 1.0)],
                                 "R-INTENSITY", "driver",
                                 inputs={"pilot": cell, "rmax_per_s": rmax,
                                         "median_op_s": p["service_s"]["median"]}))
            else:
                entries.append(K(f"corunner_{label}_levels", None, "R-INTENSITY",
                                 "driver", note="pending pilot"))
        fg = pilots.get(("e3p_fg_decode", device))
        if fg:
            entries.append(K("fg_decode_median_s", fg["service_s"]["median"],
                             "R-LAMBDA-SAT", "driver",
                             inputs={"pilot": "e3p_fg_decode"},
                             note="foreground runs saturated decode; co-runner is the swept axis"))
        entries.append(K("R", 10, "R-REPS", "driver"))
    if variant == "vqa":
        entries.append(K("R", 10, "R-REPS", "driver"))
    return entries


def derive_e5(pilots: dict, device: str) -> list[dict]:
    p = pilots.get(("e5_resnet_serial", device))
    glob_pat = ["pipeline_configs/mlperf/resnet_inference.yml"]
    entries = []
    if p:
        med = p["service_s"]["median"]
        lam = _lam(med)
        n_q = 500
        entries += [
            K("loadgen.config.rate", lam, "R-LAMBDA-BELOW-SAT", "config",
              inputs={"pilot": "e5_resnet_serial", "median_service_s": med,
                      "rho": RHO}, applies_to=glob_pat,
              verification="arrival trace: realized inter-arrivals ~ Exp(λ); 0 blocked puts"),
            K("loadgen.max_queries", n_q, "R-NTIMING", "config", applies_to=glob_pat,
              note="Server scenario cell; other scenarios get their own configs"),
            K("loadgen.queue_depth", n_q, "R-QDEPTH", "config", applies_to=glob_pat),
            K("loadgen.timeout", _timeout_open(n_q, lam, med), "R-TIMEOUT",
              "config", inputs={"unit": "seconds"}, applies_to=glob_pat),
            K("multistream_interval_s", round(8 * med, 4), "R-LAMBDA-BELOW-SAT",
              "driver", inputs={"n_samples": 8, "median_service_s": med},
              note="fixed-interval MultiStream: 8-sample query every interval"),
            K("offline_queue_depth", "total_samples", "R-QDEPTH", "driver",
              note="saturating scheduler: depth >= all samples enqueued at once"),
            K("warmup_k", p["warmup"]["k_fixed"], "R-WARMUP", "analysis",
              inputs={"pilot": "e5_resnet_serial", **p["warmup"]}),
        ]
    else:
        entries.append(K("loadgen.config.rate", None, "R-LAMBDA-BELOW-SAT",
                         "config", note="pending pilot"))
    entries.append(K("R", 10, "R-REPS", "driver"))
    return entries


def derive_e6(pilots: dict, device: str, variant: str) -> list[dict]:
    cell = "e6p_fg_ragserve" if variant == "rag_indexing" else "e6_fg_effnet"
    bg = pilots.get(("e6p_bg_index_rmax", device))
    cfg = ("pipeline_configs/rag_serve_plain*.yml" if variant == "rag_indexing"
           else "pipeline_configs/torchvision_inference.yml")
    p = pilots.get((cell, device))
    entries = []
    if p:
        med = p["service_s"]["median"]
        lam = _lam(med, E6_RHO)
        n_q = 100  # x R=5 -> 500 pooled foreground queries (p95 gate)
        entries += [
            K("fg.loadgen.config.rate", lam, "R-E6-HEADROOM", "config",
              inputs={"pilot": cell, "median_service_s": med, "rho": E6_RHO},
              applies_to=[cfg],
              verification="B=0 utilization 30-50%; λ identical across all B"),
            K("fg.loadgen.max_queries", n_q, "R-NTIMING", "config",
              applies_to=[cfg], note=">=500 pooled per cell at R=5 (p95 gate)"),
            K("fg.loadgen.queue_depth", n_q, "R-QDEPTH", "config", applies_to=[cfg]),
            K("fg.loadgen.timeout", _timeout_open(n_q, lam, med), "R-TIMEOUT",
              "config", inputs={"unit": "seconds"}, applies_to=[cfg]),
            K("warmup_k", p["warmup"]["k_fixed"], "R-WARMUP", "analysis",
              inputs={"pilot": cell, **p["warmup"]}),
        ]
    else:
        entries.append(K("fg.loadgen.config.rate", None, "R-E6-HEADROOM",
                         "config", note="pending pilot"))
    if variant == "rag_indexing":
        if bg:
            rmax = round(1.0 / bg["service_s"]["median"], 4)
            entries.append(K("bg_index_rmax_qps", rmax, "R-INTENSITY", "driver",
                             inputs={"pilot": "e6p_bg_index_rmax",
                                     "median_chunk_s": bg["service_s"]["median"],
                                     "docs_per_query": 32},
                             note="Stage-B fixed-interval levels = {25,50,75,100}% of this"))
        else:
            entries.append(K("bg_index_rmax_qps", None, "R-INTENSITY", "driver",
                             note="pending pilot e6p_bg_index_rmax"))
    entries += [
        K("B_sweep", [0, 1, 2], "R-PRECEDENT", "driver",
          note="minimal E6; ceiling reported, never extrapolated"),
        K("R", 5, "R-REPS", "driver"),
    ]
    return entries


def derive_e1_e2() -> dict:
    """E1/E2 knobs from committed data (zero-cost pilots), device-independent."""
    out = {}
    e1 = committed_warmup(
        "evaluation/overheads/framework_overhead/results/noop_d10_s0_mref_t0_r*.csv",
        window=5, epsilon=0.10,
        series_fn=lambda p: pl.per_query_latencies(p))
    e1_entries = [K("R", 10, "R-REPS", "driver")]
    if e1:
        k = e1["k_fixed"]
        e1_entries += [
            K("warmup_k", k, "R-WARMUP", "analysis", inputs=e1,
              note="replaces the old 1-of-101 drop"),
            K("loadgen.max_queries", 100 + k, "R-WARMUP", "driver",
              note="100 measured queries after warm-up"),
        ]
    out["e1"] = {"any": e1_entries}
    e2 = committed_warmup(
        "evaluation/overheads/modularity_overhead/results/mod_choreo_t1_dmps_r*.csv",
        window=51, epsilon=0.05, series_fn=_e2_step_series)
    e2_entries = [
        K("R", 10, "R-REPS", "driver"),
        K("max_batches", 1100, "R-PRECEDENT", "driver",
          note="one continuous epoch; canonicalizes the 400/1100 driver default split"),
    ]
    if e2:
        k_canon = max(200, e2["k_fixed"])
        e2_entries.append(K("warmup_k", k_canon, "R-WARMUP", "analysis",
                            inputs=e2,
                            note="canonical 200 unless detector demands more "
                                 "(detected on tracing-ON runs; tracing-OFF runs "
                                 "1-2 are contaminated and excluded)"))
    out["e2"] = {"any": e2_entries}
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", action="append", default=[],
                    help="e3=vqa|dose_response, e6=torchvision|rag_indexing")
    ap.add_argument("--e3-p95", choices=["raise", "drop"], default="raise")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Defaults = the APPROVED design (2026-07-13): VQA is cut; the staged
    # contention experiment (system -> mechanism zoom) uses the dose-response
    # cells and the RAG-serving-vs-indexing pair. The legacy variants remain
    # selectable only for archaeology.
    variants = {"e3": "dose_response", "e6": "rag_indexing"}
    for v in args.variant:
        k, _, val = v.partition("=")
        if k not in variants or not val:
            sys.exit(f"bad --variant {v}")
        variants[k] = val

    # Ingest pilots for every device that has results.
    pilots, pilot_records = {}, {}
    for f in sorted(RESULTS_DIR.glob("pilot_*_r*.csv")):
        m = re.match(r"pilot_(?P<cell>.+)_(?P<dev>mlx|cuda)_r\d+\.csv$", f.name)
        if not m:
            continue
        key = (m["cell"], m["dev"])
        if key in pilots:
            continue
        s = summarize_pilot(*key)
        if s:
            pilots[key] = s
            pilot_records[f"{key[0]}_{key[1]}"] = s
    devices = sorted({d for (_, d) in pilots})
    print(f"[derive] pilots found: {len(pilots)} cells across devices {devices}")

    experiments = derive_e1_e2()
    for dev in devices or ["mlx"]:
        dn = DEVICE_NAME[dev]
        experiments.setdefault("e4", {})[dn] = derive_e4(pilots, dev)
        e3 = derive_e3(pilots, dev, variants["e3"], args.e3_p95)
        if e3:
            experiments.setdefault("e3", {})[dn] = e3
        experiments.setdefault("e5", {})[dn] = derive_e5(pilots, dev)
        experiments.setdefault("e6", {})[dn] = derive_e6(pilots, dev, variants["e6"])

    env_commit = "unknown"
    for name in ("pilot_env_mlx.txt", "pilot_env_cuda.txt", "pilot_env.txt"):
        env_file = HERE / name
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("git_commit:"):
                    env_commit = line.split(":", 1)[1].strip()
            if env_commit != "unknown":
                break
    if env_commit.endswith("-dirty"):
        print("[derive] WARNING: pilots ran on a dirty tree — knob provenance is weakened")

    doc = {
        "schema_version": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_commit": env_commit,
        "variants": variants,
        "rules_doc": "see derive_knobs.py docstring (rule registry)",
        "pilots": pilot_records,
        "experiments": experiments,
    }
    text = yaml.safe_dump(doc, sort_keys=False, width=100)
    if args.dry_run:
        print(text)
        return 0
    KNOBS_PATH.write_text(text, encoding="utf-8")
    print(f"[derive] wrote {KNOBS_PATH}")
    n_pending = text.count("pending pilot")
    if n_pending:
        print(f"[derive] {n_pending} knob(s) pending pilots (other device?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

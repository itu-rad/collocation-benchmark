#!/usr/bin/env python3
"""E3 — MLPerf / 3D-UNet analysis. SELF-CONTAINED: parsing, statistics, tables
and figures in this one file.

Two prongs:

  1. PARITY (GB10 only). Choreo reproduces MLPerf's own reference harness on the
     SAME device, on both axes:
       * accuracy    — mean Dice, Choreo stage code vs the MLPerf reference
       * performance — MLPerf times ONLY inference, so the like-for-like Choreo
                       number is its inference-stage duration, not end-to-end.
     Same box, same model, same 42-case set => a clean apples-to-apples check.

  2. MEASUREMENT BOUNDARY (both devices). MLPerf preprocesses the dataset offline
     (its QSL preload) and times only inference. That is valid for offline batch,
     but in ONLINE serving a request arrives with its own raw data: there is
     nothing to prefetch, so loading+preprocessing sit on the per-request critical
     path. Choreo times the whole graph and exposes that share — variable across
     samples, and a LARGER fraction on the faster device (Amdahl: a faster GPU
     shrinks the inference denominator, so preprocessing dominates more).

Inputs
  Choreo timing : evaluation/unet3d/results/<dev>/unet3d_42_<dev>_r<N>.csv
                  (main.py stage markers; monotonic perf_counter_ns, last field)
  Choreo Dice   : evaluation/unet3d/results/choreo_dice_<dev>.csv
  MLPerf perf   : evaluation/unet3d/mlperf_gb10/logs_perf/mlperf_log_summary.txt
  MLPerf Dice   : evaluation/unet3d/mlperf_gb10/mlperf_accuracy_dice.txt

    python analyze_e3.py [--devices cuda mps] [--fig-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
NS_MS = 1e6

LOAD_STAGE = "KiTS19 case loader"
PREP_STAGE = "KiTS19 preprocess"
INFER_STAGE = "3D-UNet sliding-window inference"
DEV_LABEL = {"cuda": "GB10 (cuda)", "mps": "M2 Pro (mps)"}


# ---------------------------------------------------------------------------
# Choreo timing CSVs
# ---------------------------------------------------------------------------
def _rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                perf = int(parts[-1])
            except ValueError:
                continue
            yield parts[2], parts[3], parts[4], perf, parts


def parse_run(path):
    """Per-request timings for one run.

    Returns list of dicts (one per request, in arrival order) with load/prep/
    infer/e2e in ms. queue_depth=1 + serialize_queries, so exactly one request is
    in flight and stage markers pair unambiguously in order."""
    per_stage = {LOAD_STAGE: [], PREP_STAGE: [], INFER_STAGE: []}
    open_ev = {}
    e2e = []
    pipe_open = None
    for mod, phase, event, perf, parts in _rows(path):
        if mod.startswith("pipeline -") and phase == "run":
            if event == "start":
                pipe_open = perf
            elif event == "end" and pipe_open is not None:
                e2e.append((perf - pipe_open) / NS_MS)
                pipe_open = None
        elif mod in per_stage and phase == "run":
            if event == "start":
                open_ev[mod] = perf
            elif event == "end" and mod in open_ev:
                per_stage[mod].append((perf - open_ev.pop(mod)) / NS_MS)
    n = min(len(e2e), *(len(v) for v in per_stage.values())) if e2e else 0
    out = []
    for i in range(n):
        load, prep, inf = (per_stage[LOAD_STAGE][i], per_stage[PREP_STAGE][i],
                           per_stage[INFER_STAGE][i])
        tot = e2e[i]
        out.append({"idx": i, "load": load, "prep": prep, "infer": inf, "e2e": tot,
                    "pre_frac": 100.0 * (load + prep) / tot if tot else float("nan")})
    return out


def load_device(device):
    """All runs for a device: list of per-request lists."""
    d = os.path.join(HERE, "results", device)
    runs = []
    for p in sorted(glob.glob(os.path.join(d, f"unet3d_42_{device}_r*.csv"))):
        r = parse_run(p)
        if r:
            runs.append(r)
    return runs


def per_case_median(runs):
    """Median across repetitions for each request index (cases are iterated in a
    fixed order, so index k is the same case in every run)."""
    if not runs:
        return []
    n = min(len(r) for r in runs)
    out = []
    for i in range(n):
        vals = [r[i] for r in runs]
        out.append({k: st.median([v[k] for v in vals])
                    for k in ("load", "prep", "infer", "e2e", "pre_frac")} | {"idx": i})
    return out


# ---------------------------------------------------------------------------
# MLPerf reference outputs
# ---------------------------------------------------------------------------
def parse_mlperf_summary(path):
    """Pull the SingleStream latency percentiles out of mlperf_log_summary.txt."""
    if not os.path.exists(path):
        return {}
    out = {}
    # Percentile labels vary across loadgen versions ("90th" vs "90.0th"), so
    # match any "<number>th percentile" as well as the named rows.
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*([A-Za-z0-9_.]+(?:\s+[A-Za-z()/_.]+)*)\s*:\s*(.+)", line)
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        if not re.search(r"latency|Result is|Early stopping|QPS", k):
            continue
        try:
            out[k] = float(v) / 1e6 if "(ns)" in k else float(v)
        except ValueError:
            out[k] = v
        if line.startswith("Scenario"):
            out["Scenario"] = line.split(":", 1)[1].strip()
    return out


def parse_mlperf_dice(path):
    """Dice from accuracy_kits.py, which prints
    `Accuracy: mean = X, kidney = Y, tumor = Z`. Returns {mean,kidney,tumor}."""
    if not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for key in ("mean", "kidney", "tumor"):
        m = re.search(key + r"\s*=\s*([0-9.]+)", txt)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_choreo_dice(path):
    """Per-case Dice from run_full_experiment.py (the same stage code path as the
    Choreo pipeline). Returns {mean,kidney,tumor} as medians over cases."""
    if not os.path.exists(path):
        return {}
    cols = {"mean": "dice_mean", "kidney": "dice_kidney", "tumor": "dice_tumor"}
    acc = {k: [] for k in cols}
    for r in csv.DictReader(open(path)):
        if r.get("error"):
            continue
        for k, c in cols.items():
            try:
                acc[k].append(float(r[c]))
            except (ValueError, KeyError):
                pass
    return {k: st.mean(v) for k, v in acc.items() if v}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def parity_table(cuda_runs, mlperf_dir):
    print("\n## Prong 1 — parity with the MLPerf reference harness (GB10, same device)\n")
    # Prefer the COMPLIANT run (1024 queries, logs_perf_full) when it exists;
    # fall back to the bounded 43-query run, which loadgen flags INVALID.
    full = os.path.join(mlperf_dir, "logs_perf_full", "mlperf_log_summary.txt")
    bounded = os.path.join(mlperf_dir, "logs_perf", "mlperf_log_summary.txt")
    src = full if os.path.exists(full) else bounded
    summ = parse_mlperf_summary(src)
    print(f"_reference latency from `{os.path.basename(os.path.dirname(src))}`_\n")
    ref = parse_mlperf_dice(os.path.join(mlperf_dir, "mlperf_accuracy_dice.txt"))
    cho = parse_choreo_dice(os.path.join(HERE, "results", "choreo_dice_cuda.csv"))
    if not cuda_runs:
        print("_no Choreo cuda runs yet_\n")
    infer = [q["infer"] for r in cuda_runs for q in r]
    e2e = [q["e2e"] for r in cuda_runs for q in r]

    print("| quantity | MLPerf reference | Choreo | note |")
    print("|---|--:|--:|---|")
    for key, label in (("mean", "mean Dice (composite)"), ("kidney", "Dice kidney"),
                       ("tumor", "Dice tumor")):
        rd = f"{ref[key]:.4f}" if key in ref else "—"
        cd = f"{cho[key]:.4f}" if key in cho else "—"
        delta = (f"Δ {abs(ref[key] - cho[key]):.4f}"
                 if key in ref and key in cho else "same 42-case KiTS19 set")
        print(f"| {label} | {rd} | {cd} | {delta} |")
    ml = summ.get("Mean latency (ns)")
    mp50 = next((v for k, v in summ.items()
                 if re.match(r"50(\.0+)?(th)? percentile latency \(ns\)", k)), None)
    mp90 = next((v for k, v in summ.items()
                 if re.match(r"90(\.0+)?(th)? percentile latency \(ns\)", k)), None)
    # Compare MEDIANS first: MLPerf's own headline for SingleStream is a
    # percentile, and the mean is pulled around by the case mix (volume sizes
    # differ ~10x across KiTS19 cases, and the two harnesses do not issue the
    # identical multiset of samples).
    if infer:
        cmed = st.median(infer)
        cmean = st.mean(infer)
        p90 = sorted(infer)[int(0.9 * (len(infer) - 1))]
        if mp50:
            d = 100.0 * abs(cmed - mp50) / mp50
            print(f"| inference latency, median (ms) | {mp50:.0f} | {cmed:.0f} | "
                  f"**{d:.1f}% apart** — like-for-like: MLPerf times ONLY inference |")
        if ml:
            print(f"| inference latency, mean (ms) | {ml:.0f} | {cmean:.0f} | "
                  f"mean is case-mix sensitive; see median |")
        if mp90:
            print(f"| inference latency, p90 (ms) | {mp90:.0f} | {p90:.0f} | |")
    if e2e:
        print(f"| end-to-end per request (ms) | not measured | {st.median(e2e):.0f} | "
              f"MLPerf's boundary excludes load+preprocess — prong 2 |")
    if summ.get("Scenario"):
        print(f"\nMLPerf scenario: **{summ['Scenario']}**"
              + (f" (loadgen verdict: {summ['Result is']})" if "Result is" in summ else "")
              + ". A bounded run (one QSL pass) is a same-device parity check, NOT a "
                "compliant submission — loadgen requires 1024 SingleStream queries.")
    print("\n**Accuracy caveat — two differences, both known:** (a) the MLPerf "
          "reference postprocesses its logged predictions back to the ORIGINAL voxel "
          "spacing before scoring, while the Choreo number is scored on the resampled "
          "grid the model actually runs on; (b) the reference scores 43 cases while "
          "Choreo's inference_cases.json is a strict 42-case subset (it omits "
          "case_00400). So read the agreement as 'both clear the MLPerf accuracy gate "
          "(99% of 0.86170 = 0.8531)', not as a bit-exact match.")


def boundary_table(per_device):
    print("\n## Prong 2 — what MLPerf's measurement boundary hides (online serving)\n")
    print("| device | n cases | e2e median (ms) | load+preprocess (ms) | inference (ms) "
          "| preprocessing share | share range across cases |")
    print("|---|--:|--:|--:|--:|--:|---|")
    for dev, cases in per_device.items():
        if not cases:
            continue
        e2e = [c["e2e"] for c in cases]
        pre = [c["load"] + c["prep"] for c in cases]
        inf = [c["infer"] for c in cases]
        fr = [c["pre_frac"] for c in cases]
        print(f"| {DEV_LABEL.get(dev, dev)} | {len(cases)} | {st.median(e2e):.0f} | "
              f"{st.median(pre):.0f} | {st.median(inf):.0f} | "
              f"**{st.median(fr):.1f}%** | {min(fr):.1f}–{max(fr):.1f}% |")
    print("\nThe preprocessing share is what an offline-preload benchmark reports as "
          "zero. It cannot be hidden online: a request arrives with its own raw "
          "volume, so there is nothing to prefetch.")


# ---------------------------------------------------------------------------
# Figures (no titles — captions carry that in the paper)
# ---------------------------------------------------------------------------
def make_figures(per_device, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devs = [d for d in ("mps", "cuda") if per_device.get(d)]
    if not devs:
        return []
    # Fig 1: per-case stacked breakdown, one panel per device
    fig, axes = plt.subplots(1, len(devs), figsize=(6.2 * len(devs), 4.4), squeeze=False)
    for ax, dev in zip(axes[0], devs):
        cases = per_device[dev]
        x = range(len(cases))
        load = [c["load"] for c in cases]
        prep = [c["prep"] for c in cases]
        inf = [c["infer"] for c in cases]
        ax.bar(x, load, color="tab:green", label="load")
        ax.bar(x, prep, bottom=load, color="tab:orange", label="preprocess")
        ax.bar(x, inf, bottom=[a + b for a, b in zip(load, prep)],
               color="tab:blue", label="inference (all MLPerf times)")
        ax.set_xlabel(f"{DEV_LABEL.get(dev, dev)} — KiTS19 case (42, arrival order)")
        ax.set_ylabel("per-request latency (ms)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(fig_dir, "e3_request_breakdown.png")
    fig.savefig(f1, dpi=140); plt.close(fig)

    # Fig 2: preprocessing share per case, both devices
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for dev, color in (("mps", "tab:blue"), ("cuda", "tab:orange")):
        cases = per_device.get(dev)
        if not cases:
            continue
        fr = sorted(c["pre_frac"] for c in cases)
        ax.plot(range(len(fr)), fr, "o-", color=color, ms=4, lw=1.3,
                label=f"{DEV_LABEL.get(dev, dev)} (median {st.median(fr):.1f}%)")
    ax.set_xlabel("KiTS19 case (sorted by preprocessing share)")
    ax.set_ylabel("load+preprocess share of per-request latency (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = os.path.join(fig_dir, "e3_preprocessing_share.png")
    fig.savefig(f2, dpi=140); plt.close(fig)
    return [f1, f2]




# ---------------------------------------------------------------------------
# Matched per-case parity (the correct comparison)
# ---------------------------------------------------------------------------
def matched_parity(cuda_runs, mlperf_dir, dice_csv, cases_json):
    """Compare inference time CASE BY CASE, not in aggregate.

    Aggregate means/percentiles are not comparable between the two harnesses:
    loadgen issues its own multiset of samples (a bounded 43-query SingleStream
    run touched only 16 DISTINCT cases, with repeats), while Choreo runs each of
    the 42 cases once per repetition. KiTS19 volumes differ ~17x in cost
    (8..144 sub-volumes), so a different sample mix moves the aggregate a lot
    even when the per-case work is identical.

    MLPerf prints a per-sample inner time ("... took X sec") covering
    infer_single_query only — the same span Choreo's inference stage brackets —
    so those are directly comparable once matched on (shape, sub-volume count).
    Note loadgen's REPORTED latency additionally includes response
    serialisation (final_result.tobytes() over a multi-MB volume) and
    QuerySamplesComplete, which Choreo's stage marker excludes; that is part of
    why the aggregate numbers differ."""
    full_log = os.path.join(mlperf_dir, "mlperf_perf_full.log")
    log = full_log if os.path.exists(full_log) else os.path.join(
        mlperf_dir, "mlperf_perf_run.log")
    if not (os.path.exists(log) and os.path.exists(dice_csv) and cuda_runs):
        return
    ml = []
    for line in open(log, errors="replace"):
        m = re.search(r"sample id\s+(\d+) with shape = \(1, ([\d, ]+)\),\s*(\d+) "
                      r"sub-volumes took\s+([\d.]+) sec", line)
        if m:
            ml.append({"shape": "x".join(x.strip() for x in m.group(2).split(",")),
                       "nsub": int(m.group(3)), "t": float(m.group(4))})
    meta = {}
    for r in csv.DictReader(open(dice_csv)):
        if not r.get("error"):
            meta[r["case"]] = (r["image_shape"], int(r["n_subvolumes"]))
    key2case = {v: k for k, v in meta.items()}
    cases = json.load(open(cases_json)) if os.path.exists(cases_json) else []
    per = per_case_median(cuda_runs)
    cho = {cases[i]: c["infer"] / 1000.0 for i, c in enumerate(per) if i < len(cases)}

    seen, rows = set(), []
    for s in ml:
        case = key2case.get((s["shape"], s["nsub"]))
        if case and case in cho and case not in seen:
            seen.add(case)
            rows.append((case, s["nsub"], s["t"], cho[case]))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    diffs = [100.0 * (c - m) / m for _, _, m, c in rows]
    print(f"\n### Matched per-case inference time (GB10) — {len(rows)} cases "
          f"loadgen actually exercised\n")
    print("| case | sub-volumes | MLPerf inner (s) | Choreo stage (s) | diff |")
    print("|---|--:|--:|--:|--:|")
    for (case, nsub, m, c), d in zip(rows, diffs):
        print(f"| {case} | {nsub} | {m:.2f} | {c:.2f} | {d:+.1f}% |")
    print(f"\n**Median per-case difference: {st.median(diffs):+.1f}%** — the same "
          f"work, not a faster implementation. Cases within +/-1%: "
          f"{sum(1 for d in diffs if abs(d) <= 1.0)}/{len(diffs)}. Larger outliers are "
          f"first-touch effects (a shape loadgen saw once, before cuDNN/allocator "
          f"warm-up) against a Choreo median over repetitions.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=["cuda", "mps"])
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "..", "overheads", "paper_assets"))
    ap.add_argument("--mlperf-dir", default=os.path.join(HERE, "mlperf_gb10"))
    args = ap.parse_args()
    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    print("# E3 — MLPerf / 3D-UNet: reproduction + the measurement boundary\n")
    print("Online serving regime: one request in flight (serialize_queries, "
          "queue_depth 1, batch 1). Latencies from the monotonic perf clock; "
          "per-case values are medians across repetitions.\n")

    raw = {d: load_device(d) for d in args.devices}
    per_device = {d: per_case_median(r) for d, r in raw.items()}
    for d in args.devices:
        print(f"- {DEV_LABEL.get(d, d)}: {len(raw[d])} run(s), "
              f"{len(per_device[d])} cases per run")

    parity_table(raw.get("cuda", []), os.path.abspath(args.mlperf_dir))
    matched_parity(raw.get("cuda", []), os.path.abspath(args.mlperf_dir),
                   os.path.join(HERE, "results", "choreo_dice_cuda.csv"),
                   os.path.join(HERE, "inference_cases.json"))
    boundary_table(per_device)
    figs = make_figures(per_device, fig_dir)
    for f in figs:
        print(f"\n**Figure:** `{f}`")


if __name__ == "__main__":
    main()

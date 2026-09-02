#!/usr/bin/env python3
"""E3 — MLPerf 3D-UNet / KiTS19. SELF-CONTAINED: parsing, statistics, tables and
figures in this one file, so the .md and the .tex cannot disagree.

Everything on the Choreo side comes from SPANS. The perf configs set
`disable_logs` on the pipeline and on every stage, so a Choreo CSV from this
experiment holds two `prepare` rows and nothing else; if one holds hundreds, a
config lost a flag and the run was measuring its own logger.

Two prongs, and the order matters.

  1. PARITY, GB10 ONLY. Choreo's port against MLPerf's OWN reference harness on
     the SAME machine, on accuracy (DICE) and on performance. MLPerf times only
     inference, so the like-for-like Choreo number is its inference-stage
     duration -- NOT end-to-end. This is a same-device faithfulness check and
     must never be presented as a cross-device claim. Without it, prong 2 reads
     as a strawman.

  2. THE MEASUREMENT BOUNDARY, BOTH MACHINES. MLPerf preprocesses the dataset
     offline -- its QSL preloads .pkl files -- and times only inference. That is
     valid for offline batch. Online, a request arrives with its own raw volume,
     so there is nothing to prefetch and load+preprocess sit unavoidably on the
     per-request critical path. Choreo times the whole graph and reports the
     share MLPerf reports as zero. That share is VARIABLE across cases (KiTS19
     volumes tile into 8-144 sliding-window sub-volumes) and LARGER on the
     faster device, because GB10's GPU is far faster than the Mac's while CPU
     preprocessing is only somewhat faster.

     The claim is about the measurement BOUNDARY, not about an unoptimisable
     stage. DALI or a prefetch pipeline can hide preprocessing for offline
     batch; neither can prefetch a request that has not arrived yet.

Inputs
  Choreo timing : spans on res17 (experiment 138), runs named
                  unet3d_42_perf_<machine>_r<N>
  Choreo DICE   : results/dice_<machine>.csv        (from the accuracy pass)
  MLPerf perf   : mlperf_reference/logs_perf/mlperf_log_summary.txt
                  mlperf_reference/logs_perf/mlperf_log_trace.json  (per-sample)
  MLPerf DICE   : mlperf_reference/mlperf_accuracy_dice.txt
  Case order    : data/kits19/preprocessed_mlperf/preprocessed_files.pkl
                  (maps MLPerf's sample_idx to a case id)

    python analyze_e3.py [--machines m3pro gb10] [--fig-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NS_MS = 1e6
NS_US = 1e3

LOAD_STAGE = "KiTS19 case loader"
PREP_STAGE = "KiTS19 preprocess"
INFER_STAGE = "3D-UNet sliding-window inference"
SIZE_MARKER = f"{PREP_STAGE}.case_size"

MACHINE_LABEL = {"m3pro": "M3 Pro (mps)", "gb10": "GB10 (cuda)"}
PARITY_MACHINE = "gb10"          # prong 1 is same-device, and this is the device

# MLPerf's 3D-UNet/KiTS19 accuracy gate: 99% of the reference DICE.
MLPERF_REFERENCE_DICE = 0.86170
ACCURACY_GATE = 0.99 * MLPERF_REFERENCE_DICE

# The first repetition of a cell is slower for its WHOLE duration, so per-query
# warm-up dropping cannot remove it. Collect R+1 and drop it.
DROP_RUNS = 1
# The perf configs run TWO passes over the case list (max_queries = 2 x 42) and
# the first pass is dropped here. That is the whole warm-up handling: there is no
# per-case correction and nothing to stitch together.
#
# Why it is needed. A single-pass collection on gb10 ran its first four queries
# 60-82% above steady state -- reproducibly, in every repetition -- which put a
# +82% outlier into the matched parity comparison and pulled its mean to +7.0%
# against a median of -0.2%. m3pro showed no such transient (every position
# within 0.4%), so it is device behaviour rather than anything in the pipeline;
# the config is uniform anyway, so the method does not depend on which device it
# lands on.
#
# Why a whole pass rather than the four queries measured. The transient was
# measured at four on one device at one moment. Dropping a fixed small count
# would encode that measurement as an assumption. Dropping a full pass costs 2x
# runtime and encodes nothing -- and because the loader CYCLES the case list,
# queries 42-83 still cover every case exactly once, so no case is lost. That
# matters: the case is the independent variable of prong 2, and dropping the
# head of a single-pass run would have dropped specific cases.
WARMUP_PASSES = 1

# Only a DETECTOR for runs collected before the two-pass configs existed: the
# head-of-run transient was measured at four queries on gb10. It is not part of
# the method -- the method is dropping a whole pass -- and with two-pass configs
# no case should ever land in this window.
LEGACY_WARMUP_QUERIES = 4

_BOOT_WORK_BUDGET = 5e7


# ---------------------------------------------------------------------------
# Statistics — hierarchical bootstrap, the run as the unit of replication
# ---------------------------------------------------------------------------
def summarize(by_run, unit=NS_MS, n_boot=10000, seed=0):
    """median + 95% CI, resampling runs then queries within each chosen run."""
    arrs = [np.asarray(v, dtype=np.float64) for v in by_run.values()
            if len(v)]
    if not arrs:
        return None
    a = np.concatenate(arrs)
    n_eff = int(min(n_boot, max(1000, _BOOT_WORK_BUDGET // max(a.size, 1))))
    rng = np.random.default_rng(seed)
    R = len(arrs)
    boots = np.empty(n_eff)
    for i in range(n_eff):
        parts = [arrs[j][rng.integers(0, arrs[j].size, arrs[j].size)]
                 for j in rng.integers(0, R, R)]
        boots[i] = np.median(np.concatenate(parts))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"runs": R, "n": int(a.size),
            "median": float(np.median(a)) / unit,
            "mean": float(a.mean()) / unit,
            "p90": float(np.percentile(a, 90)) / unit,
            "ci_lo": float(lo) / unit, "ci_hi": float(hi) / unit,
            "run_medians": [float(np.median(v)) / unit for v in arrs]}


def fmt(s, prec=2):
    if not s:
        return "—"
    return f"{s['median']:.{prec}f} [{s['ci_lo']:.{prec}f}, {s['ci_hi']:.{prec}f}]"


# ---------------------------------------------------------------------------
# Choreo spans
# ---------------------------------------------------------------------------
# The six intervals a 3-stage graph yields per query. `entry`, the two hand-offs
# and `exit` are the framework moving the query between stages -- E2 measures
# them at 0.2-0.6 ms, which is noise against a 6 s query here, and they are kept
# only so the components provably sum to L_q.
COMPONENTS = ("entry", "load", "handoff_lp", "preprocess", "handoff_pi",
              "inference", "exit")
STAGE_WORK = ("load", "preprocess", "inference")


def span_runs(machine, tracking_uri=None, experiment="138"):
    """{label: run_id} for this machine's perf runs, from the tracking store.

    Paginated deliberately. Experiment 138 is the shared paper experiment and
    already holds thousands of runs across E1-E7; a single capped search_runs
    would quietly return a prefix of them, and the runs it dropped would look
    exactly like runs that were never collected -- a smaller R with nothing
    explaining the difference. Walk every page and say how many were seen.
    """
    import mlflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    c = mlflow.MlflowClient()
    prefix = f"unet3d_42_perf_{machine}_r"
    # {label: (start_time, run_id)} -- a label can legitimately appear more than
    # once, because re-collecting a cell reuses the run labels and the old
    # mlflow runs stay put. Keeping whichever came back last from the search
    # would pick by page order, i.e. arbitrarily, and could analyse a
    # superseded collection without saying so. Keep the most recent and say how
    # many were passed over.
    best, counts, seen, token = {}, {}, 0, None
    while True:
        page = c.search_runs([str(experiment)], max_results=1000, page_token=token)
        seen += len(page)
        for r in page:
            label = r.data.tags.get("mlflow.runName", "").split(" | ")[0]
            if not label.startswith(prefix):
                continue
            counts[label] = counts.get(label, 0) + 1
            prev = best.get(label)
            if prev is None or r.info.start_time > prev[0]:
                best[label] = (r.info.start_time, r.info.run_id)
        token = page.token
        if not token:
            break
    out = {lab: rid for lab, (_, rid) in best.items()}
    stale = {lab: n - 1 for lab, n in counts.items() if n > 1}
    if stale:
        print(f"  NOTE: {machine} has superseded mlflow runs under the same "
              f"label(s); the most recent of each is used: {stale}")
    if not out:
        print(f"  (no runs matching {prefix}* among {seen} runs in experiment "
              f"{experiment})")
    return out, c


def breakdown_by_run(machine, tracking_uri=None, experiment="138",
                     drop_runs=DROP_RUNS, warmup_passes=WARMUP_PASSES):
    """Per-run, per-query component durations plus each query's case size.

    Returns {run_index: {"comp": {name: [ns per query]}, "case": [...],
                         "n_sub": [...]}}.

    The components are successive instants within ONE query on ONE clock, so
    they are non-negative by construction and sum to L_q exactly. A negative one
    means the spans were mis-paired, which would silently shift every later
    query, so the run is refused rather than medianed over.
    """
    from utils.span_reader import read_dir, IncompleteTrace, COUNT_TAG
    labels, client = span_runs(machine, tracking_uri, experiment)
    if not labels:
        return {}
    runs = sorted(labels, key=lambda s: int(s.rsplit("_r", 1)[1]))
    if drop_runs:
        runs = runs[drop_runs:]
    out = {}
    for lab in runs:
        r = int(lab.rsplit("_r", 1)[1])
        # Pass the workload's own span count so the DROP check can run: radt
        # silently discards events on queue overflow and only warns, and a run
        # missing spans would otherwise analyse cleanly on whatever survived.
        tag = client.get_run(labels[lab]).data.tags.get(COUNT_TAG)
        try:
            t = read_dir(client.download_artifacts(run_id=labels[lab],
                                                   path="radt-trace"),
                         emitted=int(tag) if tag else None)
        except (IncompleteTrace, OSError, ValueError) as e:
            # One damaged artifact must not cost the whole report. Say which
            # run and why, and carry on with the rest -- silently dropping it
            # would leave a smaller R with nothing explaining the difference.
            print(f"  !! {lab}: trace unusable, run excluded — {e}")
            continue
        pq = t.by_query("pipeline query")
        pqp = t.by_query("pipeline query processed")
        ldr, ldp = t.by_query(f"{LOAD_STAGE}.run"), t.by_query(f"{LOAD_STAGE}.push_to_outputs")
        prr, prp = t.by_query(f"{PREP_STAGE}.run"), t.by_query(f"{PREP_STAGE}.push_to_outputs")
        inr, inp = t.by_query(f"{INFER_STAGE}.run"), t.by_query(f"{INFER_STAGE}.push_to_outputs")
        size = t.by_query(SIZE_MARKER)
        tables = (pq, pqp, ldr, ldp, prr, prp, inr, inp)
        qs = sorted([q for q in pq if all(q in d for d in tables)],
                    key=lambda q: pq[q].perf_start_ns)
        # Drop whole warm-up passes. The case list length is recovered from the
        # data rather than assumed, so this stays correct if the sweep changes.
        n_cases = len({size[q].attributes.get("case") for q in qs if q in size})
        if warmup_passes and n_cases and len(qs) > warmup_passes * n_cases:
            qs = qs[warmup_passes * n_cases:]
        elif warmup_passes and n_cases:
            print(f"  !! {lab}: {len(qs)} queries for {n_cases} case(s) — too "
                  f"few to drop {warmup_passes} warm-up pass(es). This run "
                  f"predates the two-pass configs; its head-of-run queries are "
                  f"NOT warm and it should be re-collected.")
        if not qs:
            continue
        P = lambda d, q: d[q].perf_start_ns
        comp = {
            "entry":      [P(ldr, q) - P(pq, q) for q in qs],
            "load":       [P(ldp, q) - P(ldr, q) for q in qs],
            "handoff_lp": [P(prr, q) - P(ldp, q) for q in qs],
            "preprocess": [P(prp, q) - P(prr, q) for q in qs],
            "handoff_pi": [P(inr, q) - P(prp, q) for q in qs],
            "inference":  [P(inp, q) - P(inr, q) for q in qs],
            "exit":       [P(pqp, q) - P(inp, q) for q in qs],
        }
        bad = {k: sum(1 for v in vs if v < 0) for k, vs in comp.items()}
        if any(bad.values()):
            print(f"  !! {lab}: NEGATIVE intervals {bad} — run excluded; the "
                  f"spans are mis-paired and a median over them would be wrong")
            continue
        # The case identity and its size come from the marker span the
        # preprocess stage emits, so each case's size is a property of ITS OWN
        # trace rather than something joined in from another run's side file.
        miss = [q for q in qs if q not in size]
        if miss:
            print(f"  !! {lab}: {len(miss)} query(s) with no {SIZE_MARKER} "
                  f"marker — case size unavailable for them")
        out[r] = {
            "comp": comp,
            "case": [size[q].attributes.get("case") if q in size else None for q in qs],
            "n_sub": [size[q].attributes.get("n_subvolumes") if q in size else None
                      for q in qs],
            "shape": [size[q].attributes.get("image_shape") if q in size else None
                      for q in qs],
        }
    return out


def per_case(bd):
    """{case: {component: [ns across runs], "n_sub": int}} — the per-case view.

    KiTS19 cases are not interchangeable: they differ by ~18x in sliding-window
    count, so pooling them and reporting one median throws away the independent
    variable. Every prong-2 claim is per case first and aggregated afterwards.
    """
    out = {}
    for r, d in bd.items():
        for i, case in enumerate(d["case"]):
            if case is None:
                continue
            e = out.setdefault(case, {c: [] for c in COMPONENTS})
            for c in COMPONENTS:
                e[c].append(d["comp"][c][i])
            n = d["n_sub"][i]
            if n is not None:
                e["n_sub"] = int(n)
            if d["shape"][i]:
                e["shape"] = d["shape"][i]
    return out


# ---------------------------------------------------------------------------
# MLPerf reference harness
# ---------------------------------------------------------------------------
def read_mlperf_summary(path):
    """Validity, query count and the latency percentiles the summary reports."""
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    def g(pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else None
    out = {
        "scenario": (re.search(r"Scenario\s*:\s*(\S+)", txt) or [None, None])[1]
                    if re.search(r"Scenario\s*:\s*(\S+)", txt) else None,
        "mode": (re.search(r"Mode\s*:\s*(\S+)", txt).group(1)
                 if re.search(r"Mode\s*:\s*(\S+)", txt) else None),
        "validity": (re.search(r"Result is\s*:\s*(\S+)", txt).group(1)
                     if re.search(r"Result is\s*:\s*(\S+)", txt) else None),
        "queries": g(r"Only processed (\d+) queries", int),
        "min_queries_needed": g(r"at least (\d+) queries", int),
        "min_ms": g(r"Min latency \(ns\)\s*:\s*(\d+)"),
        "max_ms": g(r"Max latency \(ns\)\s*:\s*(\d+)"),
        "mean_ms": g(r"Mean latency \(ns\)\s*:\s*(\d+)"),
        "p50_ms": g(r"50\.00 percentile latency \(ns\)\s*:\s*(\d+)"),
        "p90_ms": g(r"90\.00 percentile latency \(ns\)\s*:\s*(\d+)"),
        "p99_ms": g(r"99\.00 percentile latency \(ns\)\s*:\s*(\d+)"),
    }
    for k in ("min_ms", "max_ms", "mean_ms", "p50_ms", "p90_ms", "p99_ms"):
        if out[k] is not None:
            out[k] /= NS_MS
    return out


def read_mlperf_per_sample(trace_path, qsl_pkl):
    """{case: latency_ms} from the loadgen trace.

    SingleStream issues one sample per query, and each `Sample` begin event
    carries `complete_ns` (the query's latency, since issue_start_ns is 0) and
    `sample_idx`. The index is into the QSL's file list, which is the pickle the
    reference harness itself loads -- so the mapping is the harness's own, not a
    reconstruction.
    """
    if not (os.path.exists(trace_path) and os.path.exists(qsl_pkl)):
        return {}
    with open(qsl_pkl, "rb") as f:
        files = pickle.load(f)["file_list"]
    # A valid SingleStream run issues more queries than the QSL has samples, so
    # loadgen cycles it and every case appears several times (4x in the 172-query
    # reference run). Collect ALL occurrences and take the median, the same
    # statistic Choreo's side uses across its repetitions. Keeping one arbitrary
    # occurrence made the matched comparison read mean +7.1% with a +82% outlier
    # against a median of -0.1%: within-case spread is usually 0.3% but reaches
    # 70%, so the pick, not the framework, was the difference.
    occurrences = {}
    with open(trace_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("name") != "Sample" or rec.get("ph") != "b":
                continue
            a = rec.get("args") or {}
            idx, comp = a.get("sample_idx"), a.get("complete_ns")
            if idx is None or comp is None or idx >= len(files):
                continue
            occurrences.setdefault(files[idx], []).append(comp / NS_MS)
    return {case: float(np.median(v)) for case, v in occurrences.items()}


def read_mlperf_dice(path):
    """{mean, kidney, tumor} from the reference accuracy script's output."""
    if not os.path.exists(path):
        return None
    m = re.search(r"Accuracy:\s*mean\s*=\s*([\d.]+),\s*kidney\s*=\s*([\d.]+),"
                  r"\s*tumor\s*=\s*([\d.]+)",
                  open(path, encoding="utf-8", errors="replace").read())
    if not m:
        return None
    return {"mean": float(m.group(1)), "kidney": float(m.group(2)),
            "tumor": float(m.group(3))}


def read_choreo_dice(machine):
    path = os.path.join(HERE, "results", f"dice_{machine}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    k = [float(r["dice_kidney"]) for r in rows]
    t = [float(r["dice_tumor"]) for r in rows]
    m = [float(r["dice_mean"]) for r in rows]
    return {"cases": len(rows), "mean": float(np.mean(m)),
            "kidney": float(np.mean(k)), "tumor": float(np.mean(t)),
            "per_case": {r["case"]: float(r["dice_mean"]) for r in rows}}


# ---------------------------------------------------------------------------
# Prong 1 — parity
# ---------------------------------------------------------------------------
def reference_log_dir():
    """The reference run to compare against, valid-first.

    `logs_perf/` is the original bounded run and announces itself INVALID:
    user_e3.conf capped max_query_count at the QSL size, so loadgen never
    reached early stopping. `logs_perf_valid/` is the re-run with that cap
    lifted and a realistic target_latency. Prefer the valid one, and name which
    was used rather than leaving the reader to guess.
    """
    ref = os.path.join(HERE, "mlperf_reference")
    for name in ("logs_perf_valid", "logs_perf"):
        d = os.path.join(ref, name)
        if os.path.exists(os.path.join(d, "mlperf_log_summary.txt")):
            return d, name
    return os.path.join(ref, "logs_perf"), "logs_perf"


def case_order():
    """The fixed order the loader serves cases in, or [] if unavailable.

    Position in this list is what the warm-up transient tracks, so it is the
    only way to identify the affected cases after the fact.
    """
    path = os.path.join(HERE, "inference_cases.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def print_parity(machine, bd, cases):
    ref_dir = os.path.join(HERE, "mlperf_reference")
    log_dir, log_name = reference_log_dir()
    summary = read_mlperf_summary(os.path.join(log_dir, "mlperf_log_summary.txt"))
    ref_dice = read_mlperf_dice(os.path.join(ref_dir, "mlperf_accuracy_dice.txt"))
    our_dice = read_choreo_dice(machine)

    print(f"\n## Prong 1 — parity with the MLPerf reference harness ({machine} only)\n")
    print("Same machine, same model, same 42-case set. MLPerf times ONLY "
          "inference, so the like-for-like Choreo number is its inference-stage "
          "duration — not end-to-end. This is a same-device faithfulness check "
          "and is NOT a cross-device claim.\n")

    # --- accuracy ---
    print("### Accuracy (DICE)\n")
    print("| harness | cases | mean | kidney | tumor |")
    print("|---|--:|--:|--:|--:|")
    if ref_dice:
        print(f"| MLPerf reference | 43 | {ref_dice['mean']:.5f} | "
              f"{ref_dice['kidney']:.4f} | {ref_dice['tumor']:.4f} |")
    if our_dice:
        print(f"| Choreo | {our_dice['cases']} | {our_dice['mean']:.5f} | "
              f"{our_dice['kidney']:.4f} | {our_dice['tumor']:.4f} |")
    else:
        print("| Choreo | — | — | — | — |")
    print(f"\nMLPerf's accuracy gate is 99% of {MLPERF_REFERENCE_DICE:.5f} = "
          f"**{ACCURACY_GATE:.5f}**.", end=" ")
    if our_dice:
        verdict = "CLEARS" if our_dice["mean"] >= ACCURACY_GATE else "**FAILS**"
        print(f"Choreo scores {our_dice['mean']:.5f} and {verdict} it.")
    else:
        print("No Choreo DICE yet — run the accuracy pass "
              "(`collect_e3.sh <machine> 1 acc`).")
    print("\nThe reference scores 43 cases and Choreo 42: `inference_cases.json` "
          "omits case_00400. The two means are therefore over slightly different "
          "sets, which is stated rather than hidden — it is worth ~one case in "
          "43 and does not move the gate.")

    # --- performance ---
    print("\n### Performance (inference only, the part MLPerf times)\n")
    print(f"Reference run: `mlperf_reference/{log_name}/` "
          f"({summary['validity'] if summary else 'not found'}).\n")
    if not summary:
        print("No MLPerf summary found; the parity table cannot be built.")
        return
    if summary["validity"] != "VALID":
        print(f"> **The reference run on disk is `{summary['validity']}`.** "
              f"loadgen processed {summary['queries']} queries and needs "
              f"{summary['min_queries_needed']} to clear early stopping. Its "
              f"percentiles are reported below because they are what exists, "
              f"but the parity claim is NOT closed until a valid reference run "
              f"replaces it.\n")
    infer = summarize({r: d["comp"]["inference"] for r, d in bd.items()}) if bd else None
    print("| harness | median (ms) | mean (ms) | p90 (ms) | min | max |")
    print("|---|--:|--:|--:|--:|--:|")
    print(f"| MLPerf reference | {summary['p50_ms']:.0f} | {summary['mean_ms']:.0f} "
          f"| {summary['p90_ms']:.0f} | {summary['min_ms']:.0f} | {summary['max_ms']:.0f} |")
    if infer:
        print(f"| Choreo (inference stage) | {infer['median']:.0f} | "
              f"{infer['mean']:.0f} | {infer['p90']:.0f} | — | — |")
        d = 100.0 * (infer["median"] - summary["p50_ms"]) / summary["p50_ms"]
        print(f"\nMedian inference latency differs by **{d:+.1f}%**.")
    else:
        print("| Choreo (inference stage) | — | — | — | — | — |")
        return

    # --- matched, per case ---
    ref_cases = read_mlperf_per_sample(
        os.path.join(log_dir, "mlperf_log_trace.json"),
        os.path.join(ROOT, "data", "kits19", "preprocessed_mlperf",
                     "preprocessed_files.pkl"))
    shared = sorted(set(ref_cases) & set(cases))
    if len(shared) < 2:
        print("\nNo per-case matching possible (the reference trace or the QSL "
              "file list is missing), so only the pooled percentiles above are "
              "comparable.")
        return
    diffs, warm = [], []
    order = case_order()
    for c in shared:
        ours = float(np.median(cases[c]["inference"])) / NS_MS
        d = 100.0 * (ours - ref_cases[c]) / ref_cases[c]
        diffs.append(d)
        if order and c in order and order.index(c) < LEGACY_WARMUP_QUERIES:
            warm.append((order.index(c), c, d))
    diffs = np.asarray(diffs)
    steady = np.asarray([d for c, d in zip(shared, diffs)
                         if not (order and c in order
                                 and order.index(c) < LEGACY_WARMUP_QUERIES)])
    print(f"\n**Matched per case, all {len(shared)} cases both harnesses ran.** "
          f"Choreo vs the reference on the SAME case: median difference "
          f"**{np.median(diffs):+.1f}%**, mean {diffs.mean():+.1f}%, range "
          f"{diffs.min():+.1f}% to {diffs.max():+.1f}%.")
    if len(warm):
        print(f"\n{len(warm)} case(s) came from the head of a run "
              f"({', '.join(c for _, c, _ in sorted(warm))}). With the two-pass "
              f"configs this list should be EMPTY; if it is not, the runs "
              f"analysed predate them and carry a device warm-up transient.")


# ---------------------------------------------------------------------------
# Prong 2 — the measurement boundary
# ---------------------------------------------------------------------------
def print_boundary(per_machine):
    print("\n## Prong 2 — what the measurement boundary hides\n")
    print("MLPerf's QSL preloads preprocessed `.pkl` volumes, so its timed "
          "section starts after load and preprocess have already happened. "
          "Online they cannot: the request arrives as a raw volume. Below, "
          "`MLPerf-visible` is the inference stage alone — everything MLPerf "
          "would report — and `hidden` is what its boundary excludes.\n")
    print("| machine | R | end-to-end L_q (ms) | load | preprocess | inference "
          "| framework | hidden share (95% CI) |")
    print("|---|--:|--:|--:|--:|--:|--:|---|")
    for machine, (bd, cases) in per_machine.items():
        if not bd:
            continue
        s = {c: summarize({r: d["comp"][c] for r, d in bd.items()})
             for c in COMPONENTS}
        lq = summarize({r: [sum(v) for v in zip(*(d["comp"][c] for c in COMPONENTS))]
                        for r, d in bd.items()})
        fw = sum(s[c]["median"] for c in ("entry", "handoff_lp", "handoff_pi", "exit"))
        # The share is the median of PER-QUERY shares, not the ratio of two
        # pooled medians. Those differ, and not slightly: the cases span an 18x
        # size range, so the median preprocess and the median L_q belong to
        # different cases and their ratio describes no request that ever ran.
        share = summarize({r: [100.0 * (c_l + c_p) / tot
                               for c_l, c_p, tot in zip(
                                   d["comp"]["load"], d["comp"]["preprocess"],
                                   [sum(v) for v in zip(*(d["comp"][c]
                                                          for c in COMPONENTS))])]
                           for r, d in bd.items()}, unit=1.0)
        print(f"| {MACHINE_LABEL.get(machine, machine)} | {lq['runs']} | "
              f"{lq['median']:.0f} | {s['load']['median']:.0f} | "
              f"{s['preprocess']['median']:.0f} | {s['inference']['median']:.0f} | "
              f"{fw:.1f} | **{share['median']:.1f}%** "
              f"[{share['ci_lo']:.1f}, {share['ci_hi']:.1f}] |")
    print("\n(milliseconds, medians over all cases and repetitions. `framework` "
          "is entry + the two hand-offs + exit — the sub-millisecond scaffolding "
          "E2 measures directly; it is listed so the components provably sum to "
          "L_q, not because it is interesting at this scale.\n"
          "\n`load` is near-zero by design and that is not an error: "
          "KiTS19CaseLoader emits only the case id and its file paths. The "
          "actual read, decompress, resample, normalize and pad all happen "
          "inside `preprocess`, which is where MLPerf's offline QSL does them "
          "too. The hidden share is therefore essentially the preprocess stage.)")

    print("\n### The share is not one number — it varies by case\n")
    print("| machine | median | p25 | p75 | min | max | cases |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for machine, (bd, cases) in per_machine.items():
        if not cases:
            continue
        sh = []
        for c, d in cases.items():
            tot = sum(float(np.median(d[k])) for k in COMPONENTS)
            hid = float(np.median(d["load"])) + float(np.median(d["preprocess"]))
            if tot:
                sh.append(100.0 * hid / tot)
        if not sh:
            continue
        sh = np.asarray(sh)
        print(f"| {MACHINE_LABEL.get(machine, machine)} | {np.median(sh):.1f}% | "
              f"{np.percentile(sh, 25):.1f}% | {np.percentile(sh, 75):.1f}% | "
              f"{sh.min():.1f}% | {sh.max():.1f}% | {len(sh)} |")
    print("\nThe MEDIAN is the number to quote. The maximum is the endpoint of a "
          "range, not a representative case, and quoting it alone is the "
          "objection this table exists to answer.")
    print(f"\nIf any case above sat in the first {LEGACY_WARMUP_QUERIES} "
          f"queries of a run, that run predates the two-pass configs and "
          f"carries a device warm-up transient; re-collect rather than "
          f"correcting after the fact.")

    print("\n### Why: the two stages scale with different things\n")
    print("| machine | preprocess (ms) | inference (ms) | sub-volumes (median) "
          "| ms per sub-volume (per case, then median) |")
    print("|---|--:|--:|--:|--:|")
    for machine, (bd, cases) in per_machine.items():
        if not cases:
            continue
        pre = np.median([float(np.median(d["preprocess"])) for d in cases.values()]) / NS_MS
        inf = np.median([float(np.median(d["inference"])) for d in cases.values()]) / NS_MS
        ns = [d["n_sub"] for d in cases.values() if "n_sub" in d]
        n = float(np.median(ns)) if ns else float("nan")
        # Per case, then median -- not median(inference)/median(n), which pairs
        # numbers from two different cases.
        per_sub = [float(np.median(d["inference"])) / NS_MS / d["n_sub"]
                   for d in cases.values() if d.get("n_sub")]
        print(f"| {MACHINE_LABEL.get(machine, machine)} | {pre:.0f} | {inf:.0f} "
              f"| {n:.0f} | {np.median(per_sub):.0f} |")
    ms = [m for m in per_machine if per_machine[m][1]]
    if len(ms) == 2:
        a, b = ms
        def med(m, k):
            return np.median([float(np.median(d[k])) for d in per_machine[m][1].values()])
        print(f"\nSpeedup from {a} to {b}: inference "
              f"**{med(a, 'inference') / med(b, 'inference'):.1f}x**, preprocess "
              f"**{med(a, 'preprocess') / med(b, 'preprocess'):.1f}x**. The share "
              f"the boundary hides is larger wherever the accelerator pulls "
              f"further ahead of the CPU — which is the honest framing. It is "
              f"NOT a claim that the two machines have equal CPUs.")


def print_size_relation(per_machine):
    """The variable-size claim, with a model the data actually supports.

    The obvious model is `share = P / (P + S*n)` -- preprocessing a fixed
    per-volume cost, inference proportional to the sub-volume count. It is
    WRONG here, and it was worth finding out why rather than reporting its fit.

    Inference is proportional, cleanly: rho = +0.99/+0.98 against n, and a
    straight-line fit recovers a slope that matches the directly measured cost
    per sub-volume. But preprocessing is NOT constant. It rises with n too
    (rho = +0.77 on m3pro, +0.60 on gb10, over a 15x and 11x range), because a
    volume with more sub-volumes is a physically bigger volume and takes longer
    to read, resample and pad. Forcing P constant made the linearised fit
    absorb that growth into S, which came out 4.8x below the measured value.

    So both terms are affine in n:

        share(n) = (P0 + P1*n) / (P0 + P1*n + S0 + S1*n)

    which does not decay to zero. It falls towards **P1 / (P1 + S1)**, a floor
    set by the ratio of the two slopes. That is a stronger claim than the one
    the wrong model made: the cost MLPerf's boundary hides does not amortise
    away on large inputs, it converges to a fixed share of every request.
    """
    print("\n### Hidden share against case size\n")
    print("| machine | n range | share at min n | share at max n | Spearman rho "
          "| preprocess (ms) | inference (ms) | asymptotic share |")
    print("|---|--:|--:|--:|--:|---|---|--:|")
    for machine, (bd, cases) in per_machine.items():
        pts = [(d["n_sub"],
                float(np.median(d["load"])) + float(np.median(d["preprocess"])),
                float(np.median(d["inference"])),
                sum(float(np.median(d[k])) for k in COMPONENTS))
               for d in cases.values() if "n_sub" in d]
        if len(pts) < 4:
            continue
        n = np.array([p[0] for p in pts], dtype=float)
        hid = np.array([p[1] for p in pts]) / NS_MS
        inf = np.array([p[2] for p in pts]) / NS_MS
        tot = np.array([p[3] for p in pts]) / NS_MS
        frac = 100.0 * hid / tot
        order = np.argsort(n)
        rn = np.argsort(np.argsort(n)).astype(float)
        rf = np.argsort(np.argsort(frac)).astype(float)
        rho = float(np.corrcoef(rn, rf)[0, 1])
        P1, P0 = np.polyfit(n, hid, 1)
        S1, S0 = np.polyfit(n, inf, 1)
        asym = 100.0 * P1 / (P1 + S1)
        print(f"| {MACHINE_LABEL.get(machine, machine)} | {n.min():.0f}–{n.max():.0f} "
              f"| {frac[order][0]:.1f}% | {frac[order][-1]:.1f}% | {rho:+.2f} "
              f"| {P1:.1f}n + {P0:.0f} | {S1:.0f}n + {S0:+.0f} "
              f"| **{asym:.1f}%** |")
    print("\n(`share at min/max n` are the individual cases at the ends of the size "
          "range, not a fit. rho is Spearman's rank correlation between sub-volume "
          "count and hidden share. The two rightmost columns are straight-line fits "
          "in n, in milliseconds.\n"
          "\nBOTH stages grow with n. Inference is proportional and its fitted slope "
          "matches the directly measured cost per sub-volume. Preprocessing also "
          "rises -- a volume with more sub-volumes is a bigger volume, and reading, "
          "resampling and padding it costs more -- so the share does NOT decay to "
          "zero. It falls towards `P1 / (P1 + S1)`, the ratio of the two slopes, "
          "which is the `asymptotic share` column. The hidden cost does not amortise "
          "away on large inputs; it converges to a fixed fraction of every request.\n"
          "\nAn earlier version of this table fitted `share = P / (P + S*n)` with P "
          "constant. That model is refuted by the data -- preprocessing correlates "
          "with n at rho = +0.77 (m3pro) and +0.60 (gb10) -- and its fitted S came "
          "out 4.8x below the measured cost per sub-volume, which is what exposed it.)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_breakdown_figure(per_machine, fig_dir):
    """Per-request latency, stacked, one bar per case, ordered by size.

    This is the figure that makes the point: the inference band is what MLPerf
    reports and the load+preprocess bands are what its boundary excludes, and
    the excluded fraction visibly shrinks as the case gets bigger.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = [m for m in per_machine if per_machine[m][1]]
    if not ms:
        return None
    parts = [("load", "tab:blue"), ("preprocess", "tab:orange"),
             ("inference", "tab:green")]
    fig, ax = plt.subplots(1, len(ms), figsize=(6.2 * len(ms), 4.2), squeeze=False)
    ax = ax[0]
    for i, machine in enumerate(ms):
        cases = per_machine[machine][1]
        items = sorted([(d.get("n_sub", 0), c, d) for c, d in cases.items()])
        x = np.arange(len(items))
        bottom = np.zeros(len(items))
        for name, color in parts:
            v = np.array([float(np.median(d[name])) / NS_MS for _, _, d in items])
            ax[i].bar(x, v, bottom=bottom, color=color, width=0.9,
                      label=name + (" (MLPerf-visible)" if name == "inference"
                                    else " (hidden by MLPerf)"))
            bottom += v
        ax[i].set_xlabel(f"case, ordered by sub-volume count — "
                         f"{MACHINE_LABEL.get(machine, machine)}")
        ax[i].set_ylabel("per-request latency (ms)")
        ax[i].set_xticks(x[::4])
        ax[i].set_xticklabels([f"{items[j][0]}" for j in range(0, len(items), 4)],
                              fontsize=7)
        ax[i].grid(alpha=0.3, axis="y")
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(fig_dir, "e3_request_breakdown.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_share_figure(per_machine, fig_dir):
    """The hidden share per case, grouped by machine.

    Same shape as the request-breakdown figure: one column position per case,
    ordered by sub-volume count, with the machines side by side so a reader can
    read one case across both.

    The asymptotic share each machine tends to -- P1/(P1+S1), the ratio of the
    two stages' slopes in n -- is reported in the size-relation TABLE rather
    than drawn here, so the figure shows measurements and nothing fitted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"m3pro": "tab:blue", "gb10": "tab:orange"}
    machines = [m for m in per_machine if per_machine[m][1]]
    if not machines:
        return None

    # Order cases by sub-volume count, using whichever machine has them. The
    # count is a property of the case, not of the machine, so the two agree.
    order, seen = [], {}
    for m in machines:
        for case, d in per_machine[m][1].items():
            if "n_sub" in d:
                seen[case] = d["n_sub"]
    order = sorted(seen, key=lambda c: (seen[c], c))
    if not order:
        return None

    def share_of(d):
        tot = sum(float(np.median(d[k])) for k in COMPONENTS)
        hid = float(np.median(d["load"])) + float(np.median(d["preprocess"]))
        return 100.0 * hid / tot if tot else float("nan")

    x = np.arange(len(order))
    width = 0.8 / len(machines)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.28 * len(order)), 4.4))
    for i, m in enumerate(machines):
        cases = per_machine[m][1]
        vals = [share_of(cases[c]) if c in cases else np.nan for c in order]
        ax.bar(x + (i - (len(machines) - 1) / 2) * width, vals, width,
               color=style.get(m, "tab:green"),
               label=MACHINE_LABEL.get(m, m))
    ax.set_xlabel("case, ordered by sub-volume count (label = sub-volumes)")
    ax.set_ylabel("share of per-request latency\nMLPerf's boundary hides (%)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(seen[order[j]]) for j in range(0, len(order), 2)],
                       fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(fig_dir, "e3_preprocessing_share.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main():
    global DROP_RUNS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", nargs="+", default=["m3pro", "gb10"])
    ap.add_argument("--experiment", default=os.environ.get("E3_EXPERIMENT", "138"))
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    ap.add_argument("--drop-runs", type=int, default=DROP_RUNS,
                    help="discard the first N repetitions entirely. Defaults to "
                         "1 and should stay there: the first repetition is "
                         "slower for its WHOLE duration.")
    ap.add_argument("--warmup-passes", type=int, default=WARMUP_PASSES,
                    help="whole passes over the case list dropped at the head of "
                         "each run (default 1; the perf configs collect 2)")
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "paper_assets"))
    args = ap.parse_args()
    DROP_RUNS = args.drop_runs

    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    print("# E3 — MLPerf 3D-UNet / KiTS19\n")
    print("Every Choreo number below comes from SPANS. The perf configs set "
          "`disable_logs` on the pipeline and on all three stages, so nothing "
          "is written per query and no serialisation sits inside the measured "
          "graph.\n")
    print(f"Repetitions dropped as system warm-up: {DROP_RUNS}. Whole passes "
          f"over the case list dropped at the head of each run: "
          f"{args.warmup_passes} — the perf configs collect two passes so every "
          f"case is measured on a warm device, and the loader cycles, so the "
          f"surviving pass still covers every case exactly once. CIs are a "
          f"hierarchical bootstrap with the run as the unit of replication.\n")
    print("**E3 runs UNPINNED on both machines**, unlike E1 and E2. Pinning "
          "throttles CPU preprocessing while leaving GPU inference untouched, "
          "which would inflate the very ratio this experiment reports. "
          "`collect_e3.sh` refuses a `PIN`.\n")

    per_machine = {}
    for machine in args.machines:
        bd = breakdown_by_run(machine, args.tracking_uri, args.experiment,
                              args.drop_runs, args.warmup_passes)
        cases = per_case(bd) if bd else {}
        per_machine[machine] = (bd, cases)
        print(f"\n# ===== {MACHINE_LABEL.get(machine, machine)} "
              f"({len(bd)} run(s), {len(cases)} case(s)) =====")
        if not bd:
            print(f"\nNo perf runs found for {machine} in experiment "
                  f"{args.experiment}.")

    if PARITY_MACHINE in per_machine:
        bd, cases = per_machine[PARITY_MACHINE]
        print_parity(PARITY_MACHINE, bd, cases)

    if any(bd for bd, _ in per_machine.values()):
        print_boundary({m: v for m, v in per_machine.items() if v[0]})
        print_size_relation({m: v for m, v in per_machine.items() if v[1]})
        for f in (make_breakdown_figure(per_machine, fig_dir),
                  make_share_figure(per_machine, fig_dir)):
            if f:
                print(f"\n**Figure:** `{f}`")


if __name__ == "__main__":
    main()

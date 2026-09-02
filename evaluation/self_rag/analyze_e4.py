#!/usr/bin/env python3
"""E4 — Self-RAG decomposition & the prefill/decode split. SELF-CONTAINED.

The claim. Decomposing an agentic RAG pipeline is not simply "more overhead": it
rebalances work between two phases with DIFFERENT bottlenecks.

  prefill (TTFT)  — compute-bound: it processes the whole prompt at once
  decode          — memory-bandwidth-bound: one token at a time, weight-limited

The two DUTs have similar memory bandwidth but a large compute gap (GB10 GPU >>
M2 GPU). So the memory-bound decode should be roughly device-invariant while the
compute-bound prefill is not — and therefore the arm that wins can FLIP between
devices. A throughput-only serving benchmark cannot see this.

Markers (stages/stage.py), per LLM stage per query:
    <stage>, first_token, start|end   -> prefill window (TTFT)
    <stage>, run,         start|end   -> whole generate() call
    <stage>, n_generated_tokens, <n>  -> real token count (early EOS aware)
so decode = run_end - first_token_end, and decode tok/s = n / decode.

Each stage is a single thread pulling one query at a time, so per-stage markers
are strictly ordered and pair by sequence even under concurrent arrivals.

    python analyze_e4.py [--devices mlx cuda] [--fig-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NS_MS = 1e6
# Machines, and the result directories each one's data lives in.
#
# Results were originally filed under the ENGINE token (mlx/cuda); they are now
# filed under the MACHINE token, because the engine does not identify the
# hardware and in one case actively misled: the `mlx` tree is 16 GB M2 Pro data,
# collected before the current 24 GB m3pro existed. It is deliberately NOT read
# here -- see the provenance note in self_rag.md. `cuda` is genuine gb10 and is
# still read, since section 5.1 reuses those runs for latency and quality.
DEV_LABEL = {"m3pro": "m3pro (Apple M3 Pro, 24 GB)",
             "gb10": "gb10 (NVIDIA GB10, 120 GB)",
             "mlx": "M2 Pro (mlx) -- superseded",
             "cuda": "GB10 (cuda)"}
MACHINE_DIRS = {"m3pro": ["m3pro"], "gb10": ["gb10", "cuda"]}
ARM_ORDER = ["monolith", "monolith_4b", "decomposed", "decomposed_shared"]


def parse_label(path):
    """e4_<task>_<arm>_<device>_r<N>.csv -> dict(task, arm, device, run)."""
    m = re.match(r"^e4_(factoid|multihop)_(.+)_(m3pro|gb10|mlx|cuda)_r(\d+)\.csv$",
                 os.path.basename(path))
    if not m:
        return None
    return {"task": m.group(1), "arm": m.group(2), "device": m.group(3),
            "run": int(m.group(4)), "path": path}


def parse_run(path):
    """Per-LLM-stage, per-query prefill/decode/tokens.

    Returns {stage_name: [ {prefill_ms, decode_ms, total_ms, tokens}, ... ]}."""
    ev = defaultdict(list)
    for line in open(path, "r", encoding="utf-8"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        stage, phase, val = p[2], p[3], p[4]
        try:
            perf = int(p[-1])
        except ValueError:
            continue
        if phase in ("run", "first_token") and val in ("start", "end"):
            ev[stage].append((perf, phase, val, None))
        elif phase == "n_generated_tokens":
            try:
                ev[stage].append((perf, phase, None, int(val)))
            except ValueError:
                pass

    out = {}
    for stage, rows in ev.items():
        rows.sort(key=lambda r: r[0])
        cur, res = {}, []
        for perf, phase, val, n in rows:
            if phase == "run" and val == "start":
                cur = {"run_start": perf}
            elif phase == "first_token" and val == "start" and cur:
                cur["ft_start"] = perf
            elif phase == "first_token" and val == "end" and cur:
                cur["ft_end"] = perf
            elif phase == "n_generated_tokens" and cur:
                cur["tokens"] = n
            elif phase == "run" and val == "end" and cur.get("run_start"):
                if "ft_start" in cur and "ft_end" in cur:
                    prefill = (cur["ft_end"] - cur["ft_start"]) / NS_MS
                    decode = (perf - cur["ft_end"]) / NS_MS
                    res.append({"prefill_ms": prefill, "decode_ms": decode,
                                "total_ms": (perf - cur["run_start"]) / NS_MS,
                                "tokens": cur.get("tokens")})
                cur = {}
        if res:
            out[stage] = res
    return out


def parse_inflight(path):
    """How many queries were ACTUALLY in flight while this run was measured.

    prefill is a per-query quantity, so it is only meaningful when one query is
    in flight: two queries sharing the GPU inflate each other's TTFT and the
    measured prefill then describes the contention, not the phase. An earlier
    E4 round was invalidated for exactly that, and the re-collection asserts it
    is fixed (loadgen queue_depth 1, serialize_queries true). This function
    measures it instead of taking the config's word for it.

    The pipeline-level rows bracket each query:
        wall, <pipeline>, pipeline - <split>, run, start|end, <query_id>, ...
    with the last field a monotonic perf_counter_ns stamp. Overlaying those
    [start, end] windows and sweeping gives the count in flight at every
    instant. A correctly serialized run must show max in flight == 1.

    Returns {max, mean, queries, unclosed, span_s}; mean is time-weighted
    (the in-flight count integrated over the measured span, divided by it), so
    a brief burst cannot masquerade as sustained concurrency and vice versa."""
    open_at, windows = {}, []
    for line in open(path, "r", encoding="utf-8"):
        p = [x.strip() for x in line.split(",")]
        # the pipeline NAME contains spaces but no commas, so the field layout
        # is stable and the perf stamp is always the last field
        if len(p) < 6 or not p[2].startswith("pipeline -") or p[3] != "run":
            continue
        val, qid = p[4], p[5]
        try:
            perf = int(p[-1])
        except ValueError:
            continue
        if val == "start":
            open_at[qid] = perf
        elif val == "end" and qid in open_at:
            windows.append((open_at.pop(qid), perf))
    if not windows:
        return {"max": 0, "mean": float("nan"), "queries": 0,
                "unclosed": len(open_at), "span_s": 0.0}

    # ends before starts at an identical stamp: a query handed off to the next
    # one at the same instant is serial, not two in flight (-1 sorts before +1)
    ev = sorted([(s, 1) for s, _ in windows] + [(e, -1) for _, e in windows])
    cur = mx = 0
    area = 0.0
    prev = ev[0][0]
    for t, d in ev:
        area += cur * (t - prev)
        prev = t
        cur += d
        mx = max(mx, cur)
    span = ev[-1][0] - ev[0][0]
    # unclosed = queries the loadgen started but that never finished before the
    # run was cut off. They are left OUT of the windows, so both figures are
    # lower bounds on the true concurrency — a run flagged here was contended
    # at least as badly as reported.
    return {"max": mx, "mean": (area / span if span else float("nan")),
            "queries": len(windows), "unclosed": len(open_at),
            "span_s": span / 1e9}


def load(results_dir, machine):
    """Load every run for a machine, across each directory its data lives in."""
    runs = []
    for sub in MACHINE_DIRS.get(machine, [machine]):
        for p in sorted(glob.glob(os.path.join(results_dir, sub, "e4_*.csv"))):
            meta = parse_label(p)
            if not meta:
                continue
            stages = parse_run(p)
            if stages:
                meta["inflight"] = parse_inflight(p)
                meta["device"] = machine   # the file may carry the engine token
                runs.append((meta, stages))
    return runs


def agg_by_arm(runs):
    """{(task, arm): {prefill:[ms], decode:[ms], tok_s:[t/s], queries:n}} pooled
    over LLM stages and repetitions."""
    out = defaultdict(lambda: {"prefill": [], "decode": [], "tok_s": [],
                               "resid": [], "n": 0, "if_max": [], "if_mean": []})
    for meta, stages in runs:
        key = (meta["task"], meta["arm"])
        # concurrency is a property of the RUN, not of the call, so it is
        # appended once per run rather than once per LLM call
        infl = meta.get("inflight")
        if infl and infl["queries"]:
            out[key]["if_max"].append(infl["max"])
            out[key]["if_mean"].append(infl["mean"])
        for _stage, rows in stages.items():
            for r in rows:
                out[key]["prefill"].append(r["prefill_ms"])
                out[key]["decode"].append(r["decode_ms"])
                # Stage time in NEITHER phase: run_start -> first_token start.
                # Both backends stamp the start marker AFTER acquiring the model
                # mutex, so lock wait lands here and is invisible to a
                # prefill/decode split. It reaches ~55% of stage time on
                # contended mlx arms, so "prefill share" is a share of
                # (prefill+decode), NOT of the stage.
                out[key]["resid"].append(
                    r["total_ms"] - r["prefill_ms"] - r["decode_ms"])
                out[key]["n"] += 1
                # The decode window starts AT the first token, so it spans N-1
                # inter-token intervals, not N. Dividing by N overstates the rate
                # by N/(N-1) — 2.0x for a 2-token call — and the arms emit
                # different token counts per device, so the bias does NOT cancel
                # in the cross-device comparison.
                if r["tokens"] and r["tokens"] > 1 and r["decode_ms"] > 0:
                    out[key]["tok_s"].append(
                        1000.0 * (r["tokens"] - 1) / r["decode_ms"])
    return out


def table(per_device):
    for dev, agg in per_device.items():
        if not agg:
            continue
        print(f"\n## {DEV_LABEL.get(dev, dev)} — prefill vs decode per arm\n")
        print("| task | arm | LLM calls | prefill median (ms) | decode median (ms) "
              "| unaccounted (ms) | prefill share of p+d | decode tok/s "
              "| max in flight | mean in flight |")
        print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for (task, arm) in sorted(agg, key=lambda k: (k[0], k[1])):
            a = agg[(task, arm)]
            if not a["prefill"]:
                continue
            pf, dc = st.median(a["prefill"]), st.median(a["decode"])
            share = 100.0 * pf / (pf + dc) if (pf + dc) else float("nan")
            ts = f"{st.median(a['tok_s']):.1f}" if a["tok_s"] else "—"
            rs = st.median(a["resid"]) if a["resid"] else float("nan")
            # worst case over the cell's runs (one contended run is enough to
            # taint the pooled prefill median), against the typical load
            mx = f"{max(a['if_max'])}" if a["if_max"] else "—"
            mn = f"{st.mean(a['if_mean']):.2f}" if a["if_mean"] else "—"
            print(f"| {task} | {arm} | {a['n']} | {pf:.0f} | {dc:.0f} | "
                  f"{rs:.0f} | {share:.1f}% | {ts} | {mx} | {mn} |")


def flip_table_ci(raw):
    """Cross-device table WITH hierarchical bootstrap CIs on every ratio.

    The panel's standing objection to the previous version was that it reported
    point ratios with no uncertainty while replicates were sitting on disk. A
    ratio whose CI spans 1.0 is not evidence of a difference."""
    devs = [d for d in ("m3pro", "gb10") if raw.get(d)]
    if len(devs) < 2:
        print(f"\n_(cross-device CIs need both devices; have {devs})_\n")
        return
    a_runs, b_runs = raw[devs[0]], raw[devs[1]]
    arms = sorted({(m["task"], m["arm"]) for m, _ in a_runs}
                  & {(m["task"], m["arm"]) for m, _ in b_runs})
    if not arms:
        return
    print(f"\n## Cross-device with 95% CIs — {DEV_LABEL[devs[0]]} vs "
          f"{DEV_LABEL[devs[1]]} (run 1 dropped as warm-up)\n")
    print("| task | arm | prefill speedup [95% CI] | decode speedup [95% CI] | runs |")
    print("|---|---|--:|--:|--:|")
    for task, arm in arms:
        pa = by_run(a_runs, task, arm, "prefill_ms")
        pb = by_run(b_runs, task, arm, "prefill_ms")
        da = by_run(a_runs, task, arm, "decode_ms")
        db = by_run(b_runs, task, arm, "decode_ms")
        if not (pa and pb and da and db):
            continue
        pr = st.median([x for v in pa.values() for x in v]) / \
             st.median([x for v in pb.values() for x in v])
        dr = st.median([x for v in da.values() for x in v]) / \
             st.median([x for v in db.values() for x in v])
        plo, phi = ci_ratio(pa, pb)
        dlo, dhi = ci_ratio(da, db)
        print(f"| {task} | {arm.replace('_serial','')} | "
              f"**{pr:.2f}x** [{plo:.2f}, {phi:.2f}] | "
              f"{dr:.2f}x [{dlo:.2f}, {dhi:.2f}] | "
              f"{min(len(pa), len(pb))} |")
    print("\nDecode here is a per-CALL duration ratio; see the tok/s columns for "
          "the per-token rate, which is the speed measure.")


def flip_table(per_device):
    """The cross-device comparison: is the phase balance (and hence which arm
    wins) different on the two devices?"""
    devs = [d for d in ("m3pro", "gb10") if per_device.get(d)]
    if len(devs) < 2:
        print("\n_(cross-device flip needs both devices; only "
              f"{devs} collected so far)_\n")
        return
    a, b = per_device[devs[0]], per_device[devs[1]]
    keys = sorted(set(a) & set(b))
    if not keys:
        return
    print(f"\n## Cross-device: prefill/decode balance ({DEV_LABEL[devs[0]]} vs "
          f"{DEV_LABEL[devs[1]]})\n")
    print(f"| task | arm | prefill {devs[0]} | prefill {devs[1]} | prefill speedup "
          f"| tok/s {devs[0]} | tok/s {devs[1]} | decode speedup (per token) "
          f"| decode ratio (per call) |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for k in keys:
        pa, pb = st.median(a[k]["prefill"]), st.median(b[k]["prefill"])
        da, db = st.median(a[k]["decode"]), st.median(b[k]["decode"])
        ta = st.median(a[k]["tok_s"]) if a[k]["tok_s"] else float("nan")
        tb = st.median(b[k]["tok_s"]) if b[k]["tok_s"] else float("nan")
        print(f"| {k[0]} | {k[1]} | {pa:.0f} | {pb:.0f} | **{pa/pb:.2f}x** | "
              f"{ta:.1f} | {tb:.1f} | **{tb/ta:.2f}x** | {da/db:.2f}x |")
    print("""
**Read the per-TOKEN column, not the per-call one.** Decode duration is
(tokens x time-per-token), and the two backends do NOT emit the same number of
tokens for the same prompt under greedy decoding — MLX 4-bit and BitsAndBytes NF4
produce different outputs, e.g. decomposed_shared emits ~2.0 tokens/call on mlx
against ~4.0 on cuda. A per-call decode ratio therefore mixes decode SPEED with
how long the model chose to talk, and understates cuda by up to 2x. tok/s is the
speed measure; the per-call column is retained only to show the size of that
distortion.

If the compute-bound prefill speeds up much more than the memory-bound decode
across devices, the phase balance shifts — which is what moves the optimal
decomposition and can flip which arm wins.""")




def role_table(per_device):
    """Per-ROLE prefill/decode. This is the mechanism behind the flip: the roles
    have very different phase shapes (a grader reads long documents and emits one
    token -> almost pure prefill; a generator emits a long answer -> decode-heavy),
    so decomposing changes the MIX of compute-bound vs memory-bound work, not just
    the amount. Pooling roles together hides exactly that."""
    for dev, runs in per_device.items():
        if not runs:
            continue
        agg = defaultdict(lambda: {"prefill": [], "decode": [], "tok": []})
        for meta, stages in runs:
            for stage, rows in stages.items():
                for r in rows:
                    k = (meta["arm"], stage)
                    agg[k]["prefill"].append(r["prefill_ms"])
                    agg[k]["decode"].append(r["decode_ms"])
                    if r["tokens"] and r["tokens"] > 1 and r["decode_ms"] > 0:
                        agg[k]["tok"].append(
                            1000.0 * (r["tokens"] - 1) / r["decode_ms"])
        if not agg:
            continue
        print(f"\n### {DEV_LABEL.get(dev, dev)} — per-role phase shape\n")
        print("| arm | LLM role | calls | prefill median (ms) | decode median (ms) "
              "| prefill share | decode tok/s |")
        print("|---|---|--:|--:|--:|--:|--:|")
        for k in sorted(agg):
            a = agg[k]
            if not a["prefill"]:
                continue
            pf, dc = st.median(a["prefill"]), st.median(a["decode"])
            share = 100.0 * pf / (pf + dc) if (pf + dc) else float("nan")
            ts = f"{st.median(a['tok']):.1f}" if a["tok"] else "—"
            print(f"| {k[0]} | {k[1]} | {len(a['prefill'])} | {pf:.0f} | {dc:.0f} | "
                  f"{share:.1f}% | {ts} |")




# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
DROP_RUNS = 1          # first repetition discarded as system warm-up (see
                       # first-run-system-warmup: the first run of a cell is
                       # slower for its entire duration, and dropping it flipped
                       # the sign of E2's core overhead).


def by_run(runs, task, arm, key="prefill_ms"):
    """{run_id: [per-call values]} for one cell — the cluster structure the
    hierarchical bootstrap needs (the RUN is the unit of replication, not the
    call: calls within a run share a process, a cache and a thermal state)."""
    out = {}
    for meta, stages in runs:
        if meta["task"] != task or meta["arm"] != arm:
            continue
        vals = [r[key] for rows in stages.values() for r in rows
                if r.get(key) is not None]
        if vals:
            out[meta["run"]] = vals
    if DROP_RUNS:
        for r in sorted(out)[:DROP_RUNS]:
            out.pop(r, None)
    return out


def _boot_median(by_run_map, n_boot=4000, seed=0):
    """Bootstrap replicates of the pooled median: resample RUNS with
    replacement, then calls within each resampled run."""
    import random
    vecs = [v for _, v in sorted(by_run_map.items()) if v]
    if len(vecs) < 2:
        return []
    rng = random.Random(seed)
    R = len(vecs)
    out = []
    for _ in range(n_boot):
        pooled = []
        for _ in range(R):
            v = vecs[rng.randrange(R)]
            pooled.extend(rng.choices(v, k=len(v)))
        pooled.sort()
        n = len(pooled)
        out.append(pooled[n // 2] if n % 2 else 0.5 * (pooled[n // 2 - 1] + pooled[n // 2]))
    out.sort()
    return out


def ci_median(by_run_map, alpha=0.05):
    b = _boot_median(by_run_map)
    if not b:
        return (float("nan"), float("nan"))
    lo = b[int(alpha / 2 * (len(b) - 1))]
    hi = b[int((1 - alpha / 2) * (len(b) - 1))]
    return (lo, hi)


def ci_ratio(num_map, den_map, alpha=0.05):
    """CI on median(num)/median(den) with the two arms resampled independently
    (they are separate runs, so there is no pairing to preserve)."""
    a, b = _boot_median(num_map, seed=1), _boot_median(den_map, seed=2)
    if not a or not b:
        return (float("nan"), float("nan"))
    m = min(len(a), len(b))
    r = sorted(x / y for x, y in zip(a[:m], b[:m]) if y)
    if not r:
        return (float("nan"), float("nan"))
    return (r[int(alpha / 2 * (len(r) - 1))], r[int((1 - alpha / 2) * (len(r) - 1))])


def inflight_warnings(raw):
    """Name every run that was NOT serialized.

    Reported as its own block rather than only as a table column: a cell's
    pooled prefill median is only trustworthy if every run behind it ran one
    query at a time, and the table's max column collapses the runs, so the
    offending file has to be named to be actionable."""
    bad = []
    for dev, runs in raw.items():
        for meta, _stages in runs:
            infl = meta.get("inflight")
            if infl and infl["max"] > 1:
                bad.append((dev, meta, infl))
    print("\n## Measured concurrency check\n")
    if not bad:
        print("All runs show max in flight = 1: exactly one query was in "
              "flight at every instant, so prefill is a per-query measurement "
              "in every cell reported below.")
        return
    print(f"**WARNING — {len(bad)} run(s) were CONTENDED (max in flight > 1). "
          "Prefill for these runs measures contention, not TTFT:**\n")
    for dev, meta, infl in sorted(bad, key=lambda b: (b[0], -b[2]["max"])):
        extra = (f", {infl['unclosed']} started-but-unfinished queries excluded"
                 if infl["unclosed"] else "")
        print(f"- `{os.path.basename(meta['path'])}` ({dev}): max in flight "
              f"{infl['max']}, mean in flight {infl['mean']:.2f} over "
              f"{infl['span_s']:.0f} s, {infl['queries']} queries{extra}")


def make_figure(per_device, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    devs = [d for d in ("m3pro", "gb10") if per_device.get(d)]
    if not devs:
        return None
    fig, axes = plt.subplots(1, len(devs), figsize=(6.4 * len(devs), 4.4), squeeze=False)
    for ax, dev in zip(axes[0], devs):
        agg = per_device[dev]
        keys = [k for k in sorted(agg, key=lambda k: (k[0], k[1])) if agg[k]["prefill"]]
        labels = [f"{t} {a}" for t, a in keys]
        pf = [st.median(agg[k]["prefill"]) for k in keys]
        dc = [st.median(agg[k]["decode"]) for k in keys]
        x = range(len(keys))
        ax.bar(x, pf, color="tab:red", label="prefill (compute-bound)")
        ax.bar(x, dc, bottom=pf, color="tab:blue", label="decode (memory-bound)")
        ax.set_xticks(list(x))
        # rotate: 8 arm labels collide badly at horizontal orientation
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel(f"{DEV_LABEL.get(dev, dev)} — per-query LLM stage time (ms)")
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(fig_dir, "e4_prefill_decode.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--devices", nargs="+", default=["m3pro", "gb10"],
                    help="machines to analyse (m3pro, gb10)")
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "paper_assets"))
    args = ap.parse_args()
    fig_dir = os.path.abspath(args.fig_dir); os.makedirs(fig_dir, exist_ok=True)

    print("# E4 — Self-RAG decomposition: the prefill/decode split\n")
    print("prefill = first_token(start->end) (TTFT, compute-bound); "
          "decode = run_end - first_token_end (memory-bandwidth-bound). "
          "Pooled over LLM stages and repetitions; medians.\n")
    per_device, raw = {}, {}
    for dev in args.devices:
        runs = load(args.results_dir, dev)
        print(f"- {DEV_LABEL.get(dev, dev)}: {len(runs)} run-file(s)")
        raw[dev] = runs
        per_device[dev] = agg_by_arm(runs) if runs else {}
    print("\nmax/mean in flight = queries simultaneously inside the pipeline, "
          "from the pipeline start/end markers: the max is over the cell's "
          "runs, the mean is time-weighted within a run then averaged over "
          "them. Serialized collection must show max in flight = 1.")
    inflight_warnings(raw)
    table(per_device)
    role_table(raw)
    flip_table(per_device)
    flip_table_ci(raw)
    f = make_figure(per_device, fig_dir)
    if f:
        print(f"\n**Figure:** `{f}`")


if __name__ == "__main__":
    main()

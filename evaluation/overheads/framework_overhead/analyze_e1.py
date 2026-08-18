"""E1 (framework overhead) analysis over the NEW overnight collection layout.

Reuses the established metrics in noop_lib + the two analyzers verbatim; only the
filename scheme differs (new: `noop_depth_D_size_S_mode_M_{proc|off}_{device}_rN.csv`;
`proc` = tracing ON via the bulk+proc exporter, `off` = tracing disabled). Reads
`evaluation/results/<device>/`, writes markdown to stdout + two figures.

    python analyze_e1.py [--fig-dir DIR]
"""
import argparse
import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import noop_lib as nl                         # noqa: E402
import analyze_noop_results as ann            # noqa: E402
import analyze_payload_results as apr         # noqa: E402

NEW_RE = re.compile(
    r"^noop_depth_(?P<depth>\d+)_size_(?P<size>\d+)_mode_(?P<mode>ref|copy)"
    r"_(?P<arm>proc|off)_(?P<device>mlx|cuda)_r(?P<run>\d+)\.csv$"
)


def parse_new(path):
    m = NEW_RE.match(os.path.basename(path))
    if not m:
        return None
    return {"depth": int(m["depth"]), "size": int(m["size"]), "mode": m["mode"],
            "trace": 1 if m["arm"] == "proc" else 0,
            "device": m["device"], "run": int(m["run"]), "path": path}


def load_device(device):
    runs = []
    d = os.path.join(ROOT, "evaluation", "results", device)
    for p in sorted(glob.glob(os.path.join(d, "noop_*.csv"))):
        meta = parse_new(p)
        if not meta:
            continue
        r = nl.parse_run(p)      # content parse is filename-agnostic
        r.meta = meta            # attach our metadata (trace/size/mode/depth)
        runs.append(r)
    return runs


def tracing_add_table(od_off, od_on):
    print("\n### Tracing-layer per-stage cost (proc − off)\n")
    print("| depth | O(d) off (µs) | O(d) proc (µs) | tracing add (µs) |")
    print("|------:|--------------:|---------------:|-----------------:|")
    for d in sorted(set(od_off) & set(od_on)):
        a, b = od_off[d]["median"], od_on[d]["median"]
        print(f"| {d} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")


def make_figures(per_device, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: per-query latency L_q vs depth (linearity / flat marginal cost).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, dev in zip(axes, per_device):
        runs = per_device[dev]
        for trace, color, lbl in [(0, "tab:blue", "off"), (1, "tab:red", "proc")]:
            depths, meds = [], []
            for d in sorted({r.meta["depth"] for r in runs
                             if r.meta["size"] == 0 and r.meta["mode"] == "ref"}):
                sel = nl.select(runs, depth=d, size=0, mode="ref", trace=trace)
                s = nl.summarize(nl.pool_latency_by_run(sel), nl.NS_PER_MS)
                if s["n"]:
                    depths.append(d); meds.append(s["median"])
            if depths:
                ax.plot(depths, meds, "o-", color=color, ms=4, lw=1.3, label=f"tracing {lbl}")
                lq_ns = [m * nl.NS_PER_MS for m in meds]
                slope, icpt = nl.ols_slope(depths, lq_ns)
                xs = [min(depths), max(depths)]
                ax.plot(xs, [(slope * x + icpt) / nl.NS_PER_MS for x in xs],
                        "--", color=color, lw=0.9, alpha=0.7)
                ax.annotate(f"{lbl}: {slope / nl.NS_PER_US:.1f} µs/stage",
                            xy=(0.05, 0.9 if trace else 0.8), xycoords="axes fraction",
                            color=color, fontsize=9)
        ax.set_title(f"{dev}: per-query latency vs depth")
        ax.set_xlabel("pipeline depth (stages)"); ax.set_ylabel("L_q median (ms)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("E1 — framework dispatch is linear in depth (constant marginal per-stage cost)")
    fig.tight_layout()
    f1 = os.path.join(fig_dir, "e1_depth_flatness.png")
    fig.savefig(f1, dpi=140); plt.close(fig)

    # Fig 2: per-stage self-duration vs payload (zero-copy vs deep-copy), off arm.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {"mlx": "o", "cuda": "s"}
    for dev in per_device:
        data = apr.collect(per_device[dev])
        for mode, color in [("ref", "tab:green"), ("copy", "tab:red")]:
            xs = [s for s in apr.SIZES if s in data[mode]]
            ys = [data[mode][s]["median"] for s in xs]
            xplot = [max(x, 1) for x in xs]   # 0 -> 1 byte for log axis
            ax.plot(xplot, ys, styles[dev] + "-", color=color, ms=6, lw=1.3,
                    label=f"{dev} {mode}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("payload size (bytes; 0→1 for log axis)")
    ax.set_ylabel("per-stage self-duration median (µs)")
    ax.set_title("E1 — reference passing is O(1) in payload; deep-copy is O(payload)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = os.path.join(fig_dir, "e1_payload_zero_copy.png")
    fig.savefig(f2, dpi=140); plt.close(fig)
    return f1, f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "paper_assets"))
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    per_device = {}
    print("# E1 — Framework overhead (NoOp) — overnight collection\n")
    print(f"Warm-up epochs dropped per run: WARMUP_K={nl.WARMUP_K}. "
          "Latencies from the monotonic perf clock. CIs are hierarchical "
          "(run = unit of replication).\n")
    for dev in ("mlx", "cuda"):
        runs = load_device(dev)
        if not runs:
            print(f"## {dev}: NO runs found\n")
            continue
        per_device[dev] = runs
        n_files = len(runs)
        print(f"\n# ===== {dev} ({n_files} run-files) =====")
        od_off = ann._depth_table(nl.select(runs, trace=0), f"{dev} — tracing OFF (core dispatch)")
        od_on = ann._depth_table(nl.select(runs, trace=1), f"{dev} — tracing ON (bulk+proc)")
        tracing_add_table(od_off, od_on)
        apr.print_table(apr.collect(runs))

    if per_device:
        f1, f2 = make_figures(per_device, args.fig_dir)
        print(f"\n**Figures:** `{f1}`, `{f2}`")


if __name__ == "__main__":
    main()

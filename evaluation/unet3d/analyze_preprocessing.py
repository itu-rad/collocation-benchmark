#!/usr/bin/env python3
"""Finding A — the preprocessing MLPerf excludes is (1) sample-dependent and
(2) amplifies on the faster accelerator.

Loads the per-study 3D-UNet/KiTS19 results for both devices, prints the per-device
summary + the cross-device speedup mismatch that drives the amplification, and
(--fig) plots preprocessing share vs study size for both devices on one axis.

    python analyze_preprocessing.py [--fig preprocessing_fraction.pdf]

Data: results_mps_r1.csv (M2 Pro / mps), results_cuda_r1.csv (GB10 / cuda).
Columns: case, preprocess_s, inference_s, total_s, pre_frac_pct, n_subvolumes, ...
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    """Load one device's per-study results. Returns {} if the file is absent (e.g.
    the mps run has not been collected yet) so the analyzer degrades to whatever
    devices are present rather than crashing."""
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get("error"):
            continue
        try:
            out[r["case"]] = dict(pre=float(r["preprocess_s"]), inf=float(r["inference_s"]),
                                  frac=float(r["pre_frac_pct"]), nsub=int(r["n_subvolumes"]))
        except (ValueError, KeyError):
            continue
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def summarize(d, name):
    fr = [x["frac"] for x in d.values()]
    pre = [x["pre"] for x in d.values()]
    inf = [x["inf"] for x in d.values()]
    ns = [x["nsub"] for x in d.values()]
    print(f"\n{name}: n={len(d)}")
    print(f"  preprocess_s mean={st.mean(pre):.2f}s   inference_s mean={st.mean(inf):.2f}s")
    print(f"  pre_frac_pct median={st.median(fr):.1f}%   range=[{min(fr):.1f}, {max(fr):.1f}]%")
    print(f"  corr(pre_frac, n_subvolumes)={pearson(ns, fr):+.2f}  "
          f"(negative -> small studies are preprocessing-dominated)")
    return dict(pre=st.mean(pre), inf=st.mean(inf))


def cross_device(mac, gb):
    shared = sorted(set(mac) & set(gb))
    pm = st.mean(mac[c]["pre"] for c in shared)
    pg = st.mean(gb[c]["pre"] for c in shared)
    im = st.mean(mac[c]["inf"] for c in shared)
    ig = st.mean(gb[c]["inf"] for c in shared)
    print(f"\n=== cross-device amplification (n={len(shared)} shared studies) ===")
    print(f"  GPU  (inference) speedup Mac->GB10: {im / ig:.1f}x   ({im:.1f}s -> {ig:.1f}s)")
    print(f"  CPU  (preprocess) speedup Mac->GB10: {pm / pg:.1f}x   ({pm:.2f}s -> {pg:.2f}s)")
    print(f"  => preprocessing share Mac {pm / (pm + im) * 100:.1f}% -> GB10 {pg / (pg + ig) * 100:.1f}% "
          f"(Amdahl: the accelerator outpaces the CPU-bound stage ~{(im / ig) / (pm / pg):.1f}x)")


def stage_breakdown(mac, gb):
    """Per-stage CPU (preprocess) vs GPU (inference) latency breakdown, per device.

    This is the two-stage decomposition the paper reports: preprocessing runs on the
    CPU (resample/normalize/pad, the part MLPerf excludes) and inference runs on the
    GPU (the sliding-window forward passes, the only part MLPerf times)."""
    print("\n=== per-stage latency breakdown (CPU preprocess vs GPU inference) ===")
    print("| device | stage | hardware | mean (s) | median (s) | min–max (s) | total (s) | % of end-to-end |")
    print("|---|---|---|--:|--:|--:|--:|--:|")
    for d, name in ((mac, "M2 Pro (mps)"), (gb, "GB10 (cuda)")):
        if not d:
            continue
        pre = [v["pre"] for v in d.values()]
        inf = [v["inf"] for v in d.values()]
        tot = sum(pre) + sum(inf)
        for label, hw, xs in (("preprocess", "CPU", pre), ("inference", "GPU", inf)):
            print(f"| {name} | {label} | {hw} | {st.mean(xs):.2f} | {st.median(xs):.2f} | "
                  f"{min(xs):.2f}–{max(xs):.2f} | {sum(xs):.0f} | {100 * sum(xs) / tot:.1f}% |")


def make_stage_fig(mac, gb, out):
    """Per-study stacked bars: CPU (preprocess) + GPU (inference) time, studies sorted
    by size. Shows GPU time growing with study size while CPU stays ~flat, so small
    studies are CPU/preprocessing-dominated — on separate axes because Mac and GB10
    differ ~10x in absolute latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devs = [(d, name) for d, name in ((mac, "M2 Pro / mps"), (gb, "GB10 / cuda")) if d]
    fig, axes = plt.subplots(1, len(devs), figsize=(5.5 * len(devs), 4.5), squeeze=False)
    for ax, (d, name) in zip(axes[0], devs):
        order = sorted(d.values(), key=lambda v: v["nsub"])
        x = range(len(order))
        pre = [v["pre"] for v in order]
        inf = [v["inf"] for v in order]
        ax.bar(x, pre, color="tab:orange", label="preprocess (CPU)")
        ax.bar(x, inf, bottom=pre, color="tab:blue", label="inference (GPU)")
        ax.set_title(name)
        ax.set_xlabel("study (sorted by subvolume count →)")
        ax.set_ylabel("latency (s)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Per-study stage latency — GPU inference scales with study size, CPU preprocess stays ~flat")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    png = out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"\n[fig] wrote {out}  (+ {png})")


def make_fig(mac, gb, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d, color, marker, label in [(mac, "tab:blue", "o", "M2 Pro / mps (GPU 10x slower)"),
                                    (gb, "tab:red", "s", "GB10 / cuda (GPU 10x faster)")]:
        if not d:
            continue
        xs = [v["nsub"] for v in d.values()]
        ys = [v["frac"] for v in d.values()]
        ax.scatter(xs, ys, c=color, marker=marker, alpha=0.7, s=30, label=label, edgecolors="none")
        # mechanistic guide: frac = 100 * P / (P + S*n), P=mean preprocess, S=mean per-subvolume inference
        P = st.mean(v["pre"] for v in d.values())
        S = st.mean(v["inf"] / v["nsub"] for v in d.values())
        gx = np.linspace(min(xs), max(xs), 120)
        ax.plot(gx, 100 * P / (P + S * gx), color=color, lw=1.2, alpha=0.5)

    ax.set_xlabel("study size = sliding-window subvolumes (known from the input header)")
    ax.set_ylabel("preprocessing share of end-to-end latency (%)")
    ax.set_title("Excluded preprocessing is sample-dependent and amplifies on the faster GPU")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    png = out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"\n[fig] wrote {out}  (+ {png})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", default=None, help="write the scatter figure here (.pdf)")
    args = ap.parse_args()

    mac = load(os.path.join(HERE, "results_mps_r1.csv"))
    gb = load(os.path.join(HERE, "results_cuda_r1.csv"))
    if not mac:
        print("[note] results_mps_r1.csv not found — collect it on a Mac: "
              "run_full_experiment.py --device mps --out evaluation/unet3d/results_mps_r1.csv")
    if not gb:
        print("[note] results_cuda_r1.csv not found — collect it on GB10: "
              "run_full_experiment.py --device cuda --out evaluation/unet3d/results_cuda_r1.csv")
    if not mac and not gb:
        sys.exit("No results to analyze.")
    if mac:
        summarize(mac, "M2 Pro (mps)")
    if gb:
        summarize(gb, "GB10 (cuda)")
    if mac and gb:
        cross_device(mac, gb)  # the cross-device amplification needs both devices
    stage_breakdown(mac, gb)
    if args.fig:
        make_fig(mac, gb, args.fig)
        stem = args.fig.rsplit(".", 1)
        make_stage_fig(mac, gb, stem[0] + "_stages." + (stem[1] if len(stem) > 1 else "pdf"))


if __name__ == "__main__":
    main()

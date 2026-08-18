#!/usr/bin/env python3
"""Overhead-section figures (GB10). Data from the pinned-X925 NoOp analysis
(analyze_noop_results.py / analyze_payload_results.py). Outputs PDF (paper) + PNG (preview).

Fig 1 -- zero-copy: per-stage hand-off latency vs payload (ref O(1) vs deep-copy O(payload)).
Fig 2 -- depth-flatness: per-stage transition cost vs pipeline depth, both tracing arms.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
C_REF, C_COPY = "#1f77b4", "#d62728"
C_OFF, C_ON = "#1f77b4", "#ff7f0e"

# ---------- Fig 1: zero-copy (GB10, pinned, depth 10, tracing off) ----------
labels = ["0", "1 KiB", "1 MiB", "10 MiB"]
x = np.arange(len(labels))
ref = [41.87, 39.22, 39.73, 40.46]     # us, reference pass-by-pointer
cpy = [40.90, 43.46, 128.42, 898.19]   # us, per-hop deepcopy
COPY_SLOPE, REF_SLOPE = 81.5, 0.011    # us/MB fits

fig, ax = plt.subplots(figsize=(4.4, 3.2))
w = 0.38
ax.bar(x - w/2, ref, w, color=C_REF, label="zero-copy (ours)")
ax.bar(x + w/2, cpy, w, color=C_COPY, label="deep-copy")
ax.set_ylim(0, 950)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("payload size per query")
ax.set_ylabel("per-stage hand-off latency (µs)")
ax.grid(True, which="major", axis="y", ls=":", alpha=0.4)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_zerocopy_gb10.pdf"))
fig.savefig(os.path.join(OUT, "fig1_zerocopy_gb10.png"), dpi=150)
print("wrote fig1_zerocopy_gb10.{pdf,png}")

# ---------- Fig 2: depth-flatness (GB10, pinned) per-stage overhead L_q/depth ----------
# powers-of-2 depths only
depth = np.array([1,2,4,8,16,32,64,128])
od_off = np.array([149.97,124.16,107.16,92.43,80.45,103.75,112.34,85.01])
od_on  = np.array([2049.97,1942.06,1880.55,1750.01,1678.55,1653.61,1628.11,1659.85])

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(depth, od_on, "s-", color=C_ON, lw=1.6, ms=5, label="tracing on (+MLflow spans)")
ax.plot(depth, od_off, "o-", color=C_OFF, lw=1.6, ms=5, label="tracing off (core dispatch)")
ax.set_xscale("log")
ax.set_ylim(0, 2150)
ax.set_xlabel("pipeline depth (stages)")
ax.set_ylabel("per-stage overhead (µs)")
ax.set_xticks([1,2,4,8,16,32,64,128]); ax.set_xticklabels([1,2,4,8,16,32,64,128])
ax.grid(True, which="major", ls=":", alpha=0.35)
ax.legend(loc="center right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_depth_flatness_gb10.pdf"))
fig.savefig(os.path.join(OUT, "fig2_depth_flatness_gb10.png"), dpi=150)
print("wrote fig2_depth_flatness_gb10.{pdf,png}")

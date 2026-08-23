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
DEV_LABEL = {"mlx": "M2 Pro (mlx)", "cuda": "GB10 (cuda)"}
ARM_ORDER = ["monolith", "monolith_4b", "decomposed", "decomposed_shared"]


def parse_label(path):
    """e4_<task>_<arm>_<device>_r<N>.csv -> dict(task, arm, device, run)."""
    m = re.match(r"^e4_(factoid|multihop)_(.+)_(mlx|cuda)_r(\d+)\.csv$",
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


def load(results_dir, device):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, device, "e4_*.csv"))):
        meta = parse_label(p)
        if not meta:
            continue
        stages = parse_run(p)
        if stages:
            runs.append((meta, stages))
    return runs


def agg_by_arm(runs):
    """{(task, arm): {prefill:[ms], decode:[ms], tok_s:[t/s], queries:n}} pooled
    over LLM stages and repetitions."""
    out = defaultdict(lambda: {"prefill": [], "decode": [], "tok_s": [], "n": 0})
    for meta, stages in runs:
        key = (meta["task"], meta["arm"])
        for _stage, rows in stages.items():
            for r in rows:
                out[key]["prefill"].append(r["prefill_ms"])
                out[key]["decode"].append(r["decode_ms"])
                out[key]["n"] += 1
                if r["tokens"] and r["decode_ms"] > 0:
                    out[key]["tok_s"].append(1000.0 * r["tokens"] / r["decode_ms"])
    return out


def table(per_device):
    for dev, agg in per_device.items():
        if not agg:
            continue
        print(f"\n## {DEV_LABEL.get(dev, dev)} — prefill vs decode per arm\n")
        print("| task | arm | LLM calls | prefill median (ms) | decode median (ms) "
              "| prefill share | decode tok/s |")
        print("|---|---|--:|--:|--:|--:|--:|")
        for (task, arm) in sorted(agg, key=lambda k: (k[0], k[1])):
            a = agg[(task, arm)]
            if not a["prefill"]:
                continue
            pf, dc = st.median(a["prefill"]), st.median(a["decode"])
            share = 100.0 * pf / (pf + dc) if (pf + dc) else float("nan")
            ts = f"{st.median(a['tok_s']):.1f}" if a["tok_s"] else "—"
            print(f"| {task} | {arm} | {a['n']} | {pf:.0f} | {dc:.0f} | "
                  f"{share:.1f}% | {ts} |")


def flip_table(per_device):
    """The cross-device comparison: is the phase balance (and hence which arm
    wins) different on the two devices?"""
    devs = [d for d in ("mlx", "cuda") if per_device.get(d)]
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
                    if r["tokens"] and r["decode_ms"] > 0:
                        agg[k]["tok"].append(1000.0 * r["tokens"] / r["decode_ms"])
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


def make_figure(per_device, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    devs = [d for d in ("mlx", "cuda") if per_device.get(d)]
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
    ap.add_argument("--devices", nargs="+", default=["mlx", "cuda"])
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "..", "overheads", "paper_assets"))
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
    table(per_device)
    role_table(raw)
    flip_table(per_device)
    f = make_figure(per_device, fig_dir)
    if f:
        print(f"\n**Figure:** `{f}`")


if __name__ == "__main__":
    main()

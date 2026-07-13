#!/usr/bin/env python3
"""Post-collection verification that every knob rule actually held.

Reads collected trace CSVs + their ``*_arrivals.csv`` sidecars and checks, per
knob entry in knobs.yml that carries a ``verification`` clause:

  * R-QDEPTH        — max put-block < 5 ms (queue never filled);
  * R-LAMBDA-*      — realized mean arrival rate within 5% of intended λ;
  * R-WARMUP        — the post-k rolling median is flat (detect_warmup on the
                      collected series confirms k was sufficient).

Fills the ``verified:`` fields in knobs.yml (true / false + note) and exits
nonzero if any check fails — CI-able, and generate_knob_tables.py re-rendered
afterwards shows the populated Verification column.

    python evaluation/pilots/verify_knobs.py --traces 'evaluation/results/e4_*' --experiment e4 --device m2pro
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_lib as pl  # noqa: E402

REPO_ROOT = HERE.parent.parent
BLOCK_S = 0.005
RATE_TOL = 0.05


def verify_traces(trace_glob: str, lam: float | None, warmup_k: int | None):
    csvs = [Path(p) for p in sorted(globmod.glob(str(REPO_ROOT / trace_glob)))
            if p.endswith(".csv") and not p.endswith("_arrivals.csv")]
    if not csvs:
        return None, [f"no traces match {trace_glob}"]
    notes, ok = [], True
    for c in csvs:
        sidecar = c.with_name(c.stem + "_arrivals.csv")
        if sidecar.exists():
            arr = pl.parse_arrivals(sidecar)
            if arr.max_block_s > BLOCK_S:
                ok = False
                notes.append(f"{c.name}: {arr.blocked_puts} blocked puts "
                             f"(max {arr.max_block_s * 1000:.1f} ms) — R-QDEPTH violated")
            if lam:
                rr = arr.realized_rate()
                if rr == rr and abs(rr - lam) / lam > RATE_TOL:
                    ok = False
                    notes.append(f"{c.name}: realized rate {rr:.4f} vs λ={lam} "
                                 f"(>±{RATE_TOL:.0%}) — R-LAMBDA violated")
        elif lam:
            notes.append(f"{c.name}: no arrivals sidecar (pre-sidecar run?)")
        if warmup_k is not None:
            x = pl.per_query_latencies(c)
            if len(x) > warmup_k + 10:
                wu = pl.detect_warmup(x[warmup_k:], window=5, epsilon=0.10)
                if wu.converged and wu.k_star > 0:
                    ok = False
                    notes.append(f"{c.name}: post-k series still warming "
                                 f"(k*={wu.k_star} after drop) — R-WARMUP violated")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", required=True,
                    help="glob (repo-relative) over collected trace CSVs")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--device", required=True, choices=["m2pro", "gb10", "any"])
    args = ap.parse_args()

    knobs = pl.load_knobs()
    if not knobs:
        sys.exit("knobs.yml not found")
    entries = (knobs.get("experiments", {}).get(args.experiment, {})
               .get(args.device) or [])
    if not entries:
        sys.exit(f"no knob entries for {args.experiment}/{args.device}")

    lam = next((e["value"] for e in entries
                if e["knob"].endswith("config.rate") and e.get("value")), None)
    warmup_k = next((e["value"] for e in entries if e["knob"] == "warmup_k"), None)

    ok, notes = verify_traces(args.traces, lam, warmup_k)
    for n in notes:
        print(f"  {n}")
    if ok is None:
        return 2

    stamp = {"ok": bool(ok), "traces": args.traces,
             "checked": __import__("time").strftime("%Y-%m-%d")}
    if notes and not ok:
        stamp["notes"] = notes[:10]
    for e in entries:
        if e.get("verification"):
            e["verified"] = stamp
    pl.KNOBS_PATH.write_text(
        yaml.safe_dump(knobs, sort_keys=False, width=100), encoding="utf-8")
    print(f"[verify] {args.experiment}/{args.device}: "
          f"{'OK' if ok else 'FAILED'} — knobs.yml updated")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

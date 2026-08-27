#!/usr/bin/env python3
"""Post-collection analyzer for the staged contention experiment (Stages A-D).

Design of record: CONTENTION_EXPERIMENTS_REDESIGN.md (E3'/E6' staged form).
Consumes the curated collection outputs under evaluation/collect/results/
(evaluation/collect/results/<dev>/<label>_r<r>.csv [+ _arrivals.csv sidecars,
+ optional <label>_r<r>_bandwidth.csv AMC traces]) and emits:

  staged_per_run.csv        one tidy row per (cell, run, pipeline-role)
  staged_cell_estimates.csv cell-level estimates + hierarchical 95% CIs,
                            dose-response slopes, pairwise slope ratios
  staged_report.txt         per-stage tables + the pre-registered H1/H2
                            verdict lines (falsification wording per the
                            design doc), and the list of missing cells
  figures (--plots)         matplotlib (Agg backend; no display required)

    python evaluation/contention/analyze_staged.py --device mlx
    python evaluation/contention/analyze_staged.py --device cuda \\
        --results-dir evaluation/collect/results/cuda --plots

Statistical conventions (shared with the rest of the paper via staged_lib /
noop_lib): the RUN is the unit of replication; per-query quantiles get a
hierarchical (run-then-query) bootstrap CI; run-level scalars (throughput,
realized co-runner rate) get a run-resampling CI; dose-response slopes get a
within-level run-resampling bootstrap, and pairwise engine slope ratios are
judged against the pre-registered [2/3, 3/2] band at matched bytes/s.

Step D phase split — METHOD AND ITS LIMIT (verified against the actual
configs and stage code on 2026-07-14):

  The Stage-D foreground (configs/stage_d_*_{dev}.yml pipeline[0] "Decode 9B
  pilot", taken verbatim from pipeline_configs/pilots/decode_9b_*.yml) is
  Mock dataloader -> Decode LLM (stages.llm_mlx.Inference on mlx,
  stages.llm_huggingface.Inference on cuda). BOTH generator stages execute
  generation as a single black-box call (mlx_lm.generate at
  stages/llm_mlx/inference.py:98-104; model.generate at
  stages/llm_huggingface/inference.py:198-202) inside the one run() span, and
  the framework logs only whole-stage 'run start/end' rows per query
  (stages/stage.py:271-293). NO first-token or per-token event exists in the
  trace today. Therefore:

    * gen_dur_s      = Decode-LLM stage duration per query (trace-derived,
                       FIFO-paired stage rows);
    * per_token_s    = gen_dur_s / n_tokens. n_tokens comes from the trace's
                       per-query "n_generated_tokens" rows when the
                       instrumented generator stages produced them (early-EOS
                       correct), else falls back to the config's
                       gen_kwargs.max_tokens (256; greedy). The fallback
                       UPPER-BOUNDS the true per-token decode time by
                       prefill/n_tokens (alpaca prompts are short; bias <~
                       TTFT/256) and underestimates when the model EOSes
                       early.
    * ttft_s         = NaN — NOT DERIVABLE from current traces. H2 (TTFT flat
                       vs per-token degrading) is reported NOT EVALUABLE
                       until the generator stage logs a first-token event.
                       The parser already picks up "<stage>::prefill" /
                       "<stage>::first_token" sub-phase rows the moment such
                       instrumentation lands, and this analyzer will then
                       compute ttft_s = first_token_end - stage_start and
                       per_token_s = (stage_end - first_token_end) /
                       (n_tokens - 1) automatically.

Graceful mid-flight behavior: analyzes whatever cells exist, reports the
expected-vs-present matrix (expected = configs/stage_*_{dev}.yml x R=5).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import staged_lib as sl  # noqa: E402

CONFIG_DIR = HERE / "configs"
EXPECTED_R = 5
BAND = (2.0 / 3.0, 3.0 / 2.0)          # pre-registered slope-ratio band
KIND_ENGINE = {"stream": "cpu", "clipgpu": "gpu", "clipane": "ane",
               "indexer": "gpu"}       # embedder is GPU-placed

_FNAME_RE = re.compile(
    r"^(?P<cell>stage_(?P<st>[abcd])_(?P<mid>.+?)_(?P<dev>mlx|cuda))"
    r"(?:_(?:mlx|cuda))?_r(?P<run>\d+)"
    r"(?P<suffix>_arrivals|_outputs|_bandwidth)?\.(?:csv|jsonl)$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(results_dir: Path, device: str) -> dict[str, dict[int, dict]]:
    """{cell_stem: {run: {"traces": [paths], "arrivals": path|None,
    "bandwidth": path|None}}} — cell_stem matches the config filename stem."""
    cells: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in sorted(results_dir.glob("stage_*")):
        m = _FNAME_RE.match(p.name)
        if not m or m["dev"] != device:
            continue
        run = int(m["run"])
        entry = cells[m["cell"]].setdefault(
            run, {"traces": [], "arrivals": None, "bandwidth": None})
        if m["suffix"] == "_arrivals":
            entry["arrivals"] = p
        elif m["suffix"] == "_bandwidth":
            entry["bandwidth"] = p
        elif m["suffix"] == "_outputs":
            pass
        elif p.suffix == ".csv":
            entry["traces"].append(p)
    # AMC sidecars may still sit uncurated in evaluation/results/
    gdir = Path(sl.global_results_dir())
    for cell, runs in cells.items():
        for run, entry in runs.items():
            if entry["bandwidth"] is None:
                for cand in (gdir / f"{cell}_{device}_r{run}_bandwidth.csv",
                             gdir / f"{cell}_r{run}_bandwidth.csv"):
                    if cand.exists():
                        entry["bandwidth"] = cand
                        break
    return {c: dict(rs) for c, rs in cells.items() if any(
        e["traces"] for e in rs.values())}


def parse_cell_meta(cell: str) -> dict:
    """stage / B / kind / level_pct from the cell stem."""
    m = re.match(r"^stage_(?P<st>[abcd])_(?P<mid>.+)_(?P<dev>mlx|cuda)$", cell)
    st, mid = m["st"], m["mid"]
    meta = {"stage": st, "B": float("nan"), "kind": "", "level_pct": float("nan")}
    if st == "a":
        meta["B"] = int(mid[1:])
        meta["kind"] = "indexer" if meta["B"] else "none"
    elif st == "b":
        meta["kind"] = "indexer"
        meta["level_pct"] = int(mid[1:])
    else:
        kind, lvl = mid.rsplit("_L", 1)
        meta["kind"], meta["level_pct"] = kind, int(lvl)
    return meta


def load_cell_config(cell: str) -> dict | None:
    p = CONFIG_DIR / f"{cell}.yml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def pipeline_roles(cfg: dict | None, traces: dict) -> tuple[str | None, list[str]]:
    """(foreground name, [background names]) — configs are the source of
    truth (pipelines[0] is the foreground by generator construction); falls
    back to the 'BG ' name prefix for traces without a config."""
    names = list(traces)
    if cfg and cfg.get("pipelines"):
        fg = cfg["pipelines"][0]["name"]
        bg = [pl["name"] for pl in cfg["pipelines"][1:]]
        if fg in traces or not names:
            return fg, bg
    bg = [n for n in names if n.startswith("BG ")]
    fgs = [n for n in names if not n.startswith("BG ")]
    return (fgs[0] if fgs else None), bg


def loadgen_info(pcfg: dict) -> dict:
    lg = pcfg.get("loadgen", {})
    comp = lg.get("component", "")
    conf = lg.get("config") or {}
    offered = float("nan")
    if "MultiStream" in comp and conf.get("interval"):
        offered = 1.0 / float(conf["interval"])
    elif "Poisson" in comp and conf.get("rate"):
        offered = float(conf["rate"])
    return {"component": comp.rsplit(".", 1)[-1], "offered_qps": offered,
            "max_queries": lg.get("max_queries")}


def bg_traffic_model(pcfg: dict) -> tuple[float, str]:
    """(bytes per query, note) for a background pipeline, model-based.
    Exact for MemoryStream (stages/evaluation/memory_stream.py bytes_per_query);
    NaN for encode/index co-runners (counters are the measurement of record)."""
    for st in pcfg.get("stages", []):
        if st.get("component", "").endswith("MemoryStream"):
            c = st.get("config") or {}
            bpq = int(c.get("passes", 4)) * 3 * int(c.get("size_mb", 256)) * (1 << 20)
            return float(bpq), "stream-triad model (exact)"
    return float("nan"), "no model-based traffic estimate (use counters)"


def fg_gen_stage(pcfg: dict) -> tuple[str | None, int]:
    """(generator stage name, n_tokens) of a foreground pipeline config."""
    for st in reversed(pcfg.get("stages", [])):
        if "Inference" in st.get("component", ""):
            gk = ((st.get("config") or {}).get("model") or {}).get("gen_kwargs") or {}
            return st["name"], int(gk.get("max_tokens", 0) or 0)
    return None, 0


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def analyze_run(cell: str, meta: dict, run: int, entry: dict, cfg: dict | None,
                fg_warmup: int, bg_warmup_s: float) -> list[dict]:
    """Tidy per-run rows (one per pipeline role present in the trace)."""
    traces = sl.parse_trace_files(entry["traces"])
    fg_name, bg_names = pipeline_roles(cfg, traces)
    cfg_by_name = {pl["name"]: pl for pl in (cfg or {}).get("pipelines", [])}

    arr, arr_owner, arr_frac = None, None, 0.0
    if entry["arrivals"]:
        arr = sl.parse_arrivals(entry["arrivals"])
        if arr.n:
            arr_owner, arr_frac = sl.infer_arrivals_owner(traces, arr)

    bw_rows = sl.load_bandwidth_csv(entry["bandwidth"]) if entry["bandwidth"] else []

    rows = []
    for name, pt in traces.items():
        role = "fg" if name == fg_name else ("bg" if name in bg_names or
                                             name.startswith("BG ") else "unknown")
        pcfg = cfg_by_name.get(name, {})
        lg = loadgen_info(pcfg) if pcfg else {"component": "?", "offered_qps":
                                              float("nan"), "max_queries": None}
        t0, t1 = pt.span_wall()
        span = t1 - t0 if not math.isnan(t0) else float("nan")
        row = {"cell": cell, **meta, "run": run, "pipeline": name, "role": role,
               "n_queries": len(pt.queries), "n_completed": len(pt.completed),
               "span_s": span, "scheduler": lg["component"],
               "offered_qps": lg["offered_qps"],
               "anchor": "", "match_frac": float("nan"),
               "blocked_puts": float("nan"), "notes": ""}

        if role == "fg":
            use_arr = arr if (arr is not None and arr_owner == name) else None
            resp, anchor, diag = sl.anchored_responses(pt, use_arr)
            if arr is not None and arr_owner != name:
                row["notes"] += (f"arrivals sidecar owned by "
                                 f"'{arr_owner}' (match {arr_frac:.2f}); ")
            resp_used = resp[fg_warmup:]
            row.update({
                "anchor": anchor, "match_frac": diag["match_frac"],
                "blocked_puts": diag["blocked_puts"],
                "n_used": len(resp_used),
                "p50_resp_s": sl.median(resp_used),
                "p95_resp_s": sl.p95(resp_used) if len(resp_used) >= 20
                else float("nan"),
                "mean_resp_s": (sum(resp_used) / len(resp_used))
                if resp_used else float("nan"),
                "throughput_qps": (len(pt.completed) / span)
                if span and span > 0 else float("nan"),
                "_responses": resp_used,
            })
            # Step D phase split (see module docstring for method + limits)
            gen_stage, cfg_tok = (fg_gen_stage(pcfg) if pcfg else (None, 0))
            if meta["stage"] == "d" and gen_stage:
                durs_all = sl.stage_durations_by_query(pt, gen_stage)[fg_warmup:]
                counts = pt.stage_token_counts.get(gen_stage, [])[fg_warmup:]
                # Prefer trace-logged real token counts (early-EOS correct);
                # fall back to the config's max_tokens per query.
                durs, ntoks = [], []
                for i, d in enumerate(durs_all):
                    if math.isnan(d):
                        continue
                    n = counts[i] if i < len(counts) and counts[i] > 0 else cfg_tok
                    if n > 0:
                        durs.append(d)
                        ntoks.append(n)
                ttfts = _ttft_from_subphases(pt, gen_stage, fg_warmup)
                if durs:
                    if counts:
                        row["notes"] += "n_tokens from trace rows; "
                    if ttfts:
                        per_tok = [(d - t) / max(n - 1, 1)
                                   for d, t, n in zip(durs, ttfts, ntoks)]
                        row["ttft_s_med"] = sl.median(ttfts)
                        row["notes"] += "ttft from sub-phase rows; "
                    else:
                        per_tok = [d / n for d, n in zip(durs, ntoks)]
                        row["ttft_s_med"] = float("nan")
                        row["notes"] += ("per_token=gen_dur/n_tokens "
                                         "(no first-token event; incl. prefill); ")
                    row.update({
                        "gen_dur_s_med": sl.median(durs),
                        "per_token_s_med": sl.median(per_tok),
                        "tok_per_s_med": sl.median(
                            [n / d for n, d in zip(ntoks, durs)]),
                        "_tok_per_s": [n / d for n, d in zip(ntoks, durs)],
                        "_per_token_s": per_tok,
                        "_ttft_s": ttfts,
                    })
        elif role == "bg":
            # realized traffic over the post-warm-up window
            warm = min(bg_warmup_s, span / 4.0) if span and span > 0 else 0.0
            done = [q for q in pt.completed if q.end_wall >= t0 + warm]
            wspan = t1 - (t0 + warm)
            realized = len(done) / wspan if wspan and wspan > 0 else float("nan")
            bpq, bnote = bg_traffic_model(pcfg) if pcfg else (float("nan"), "no config")
            row.update({
                "n_used": len(done),
                "realized_qps": realized,
                "offered_bytes_ps": lg["offered_qps"] * bpq,
                "realized_bytes_ps": realized * bpq,
                "throughput_qps": realized,
                "notes": row["notes"] + bnote + "; ",
            })
            if arr is not None and arr_owner == name:
                row["match_frac"] = arr_frac
                row["notes"] += "arrivals sidecar owned by this bg; "

        if bw_rows and not math.isnan(t0):
            row.update({f"amc_{k}": v for k, v in
                        sl.bandwidth_window_stats(bw_rows, t0, t1).items()})
        rows.append(row)

    if not traces:
        rows.append({"cell": cell, **meta, "run": run, "pipeline": "",
                     "role": "empty", "notes": "trace parsed to zero pipelines"})
    return rows


def _ttft_from_subphases(pt, gen_stage: str, warmup: int) -> list[float]:
    """TTFT per query when the generator logs a prefill/first-token sub-phase
    (forward-compat path; empty today — see module docstring)."""
    for suffix in ("::prefill", "::first_token"):
        key = f"{gen_stage}{suffix}"
        if key in pt.stage_execs:
            gen = pt.stage_execs.get(gen_stage, [])
            sub = pt.stage_execs[key]
            n = min(len(gen), len(sub))
            return [(sub[i][1] - gen[i][0]) / sl.NS for i in range(n)][warmup:]
    return []


# ---------------------------------------------------------------------------
# Cell-level aggregation
# ---------------------------------------------------------------------------

def agg_cell(cell_rows: list[dict], seed: int) -> list[dict]:
    """Cell-level estimates + CIs from the tidy per-run rows of ONE cell."""
    import numpy as np
    out = []
    base = {k: cell_rows[0][k] for k in
            ("cell", "stage", "B", "kind", "level_pct")}
    fg = [r for r in cell_rows if r["role"] == "fg"]
    bg = [r for r in cell_rows if r["role"] == "bg"]

    def add(metric, est, ci, runs, unit=""):
        out.append({**base, "metric": metric, "estimate": est,
                    "ci_lo": ci[0], "ci_hi": ci[1], "n_runs": len(runs),
                    "unit": unit,
                    "run_values": ";".join(sl.fmt(v) for v in runs)})

    if fg:
        resp_vecs = [r.get("_responses", []) for r in fg]
        add("fg_p50_resp_s", sl.median([v for vec in resp_vecs for v in vec]),
            sl.hier_boot_ci(resp_vecs, np.median, seed=seed),
            [r.get("p50_resp_s", float("nan")) for r in fg], "s")
        pooled = sum(len(v) for v in resp_vecs)
        if pooled >= 500:                      # paper's p95 gate
            add("fg_p95_resp_s", sl.p95([v for vec in resp_vecs for v in vec]),
                sl.hier_boot_ci(resp_vecs, lambda a: np.percentile(a, 95),
                                seed=seed),
                [r.get("p95_resp_s", float("nan")) for r in fg], "s")
        thr = [r.get("throughput_qps", float("nan")) for r in fg]
        add("fg_throughput_qps", np.nanmean(thr) if thr else float("nan"),
            sl.run_level_ci(thr, seed=seed), thr, "1/s")
        tok_vecs = [r["_tok_per_s"] for r in fg if "_tok_per_s" in r]
        if tok_vecs:
            add("fg_tok_per_s", sl.median([v for vec in tok_vecs for v in vec]),
                sl.hier_boot_ci(tok_vecs, np.median, seed=seed),
                [r.get("tok_per_s_med", float("nan")) for r in fg], "tok/s")
            pt_vecs = [r["_per_token_s"] for r in fg if "_per_token_s" in r]
            add("fg_per_token_s", sl.median([v for vec in pt_vecs for v in vec]),
                sl.hier_boot_ci(pt_vecs, np.median, seed=seed),
                [r.get("per_token_s_med", float("nan")) for r in fg], "s")
            tt = [r.get("ttft_s_med", float("nan")) for r in fg]
            if any(not math.isnan(v) for v in tt):
                tt_vecs = [r["_ttft_s"] for r in fg if r.get("_ttft_s")]
                add("fg_ttft_s", sl.median([v for vec in tt_vecs for v in vec]),
                    sl.hier_boot_ci(tt_vecs, np.median, seed=seed), tt, "s")
    if bg:
        # realized co-runner traffic; multiple bg pipelines (stage A B=2) sum
        by_run = defaultdict(float)
        for r in bg:
            if not math.isnan(r.get("realized_qps", float("nan"))):
                by_run[r["run"]] += r["realized_qps"]
        vals = list(by_run.values())
        add("bg_realized_qps", (sum(vals) / len(vals)) if vals else float("nan"),
            sl.run_level_ci(vals, seed=seed), vals, "1/s")
        bts = [r.get("realized_bytes_ps", float("nan")) for r in bg]
        if any(not math.isnan(v) for v in bts):
            import numpy as _np
            add("bg_realized_bytes_ps", float(_np.nanmean(bts)),
                sl.run_level_ci(bts, seed=seed), bts, "B/s")
    for eng in ("cpu", "gpu", "ane", "total"):
        key = f"amc_{eng}_gbps"
        vals = [r[key] for r in fg if key in r]
        if vals:
            add(key, sum(vals) / len(vals), sl.run_level_ci(vals, seed=seed),
                vals, "GB/s")
    return out


# ---------------------------------------------------------------------------
# Dose-response slopes, knee, H1/H2
# ---------------------------------------------------------------------------

def knee_and_points(kind_rows: dict[int, list[dict]], y_key: str,
                    knee_tol: float, amc_engine: str | None,
                    amc_baseline: float):
    """Per-kind ladder -> (points [(x, [run y])], x_unit, saturated_levels).

    Knee rule (pre-registered): a ladder level is saturated when the mean
    realized co-runner rate falls short of offered by more than knee_tol;
    slopes use only levels BELOW the first saturated one. x-axis preference:
    AMC counter bytes/s (engine bucket minus isolation baseline) > model-based
    offered bytes/s (stream) > offered ops/s.
    """
    import numpy as np
    levels = sorted(kind_rows)
    saturated = []
    for lvl in levels:
        bg = [r for r in kind_rows[lvl] if r["role"] == "bg"]
        off = [r["offered_qps"] for r in bg
               if not math.isnan(r.get("offered_qps", float("nan")))]
        rea = [r.get("realized_qps", float("nan")) for r in bg]
        if off and rea and not all(math.isnan(v) for v in rea):
            if np.nanmean(rea) < (1.0 - knee_tol) * np.mean(off):
                saturated.append(lvl)
    first_sat = min(saturated) if saturated else None
    usable = [lvl for lvl in levels if first_sat is None or lvl < first_sat]

    def xs_for(lvl):
        bg = [r for r in kind_rows[lvl] if r["role"] == "bg"]
        fgr = [r for r in kind_rows[lvl] if r["role"] == "fg"]
        # 1) counter-based bytes/s
        if amc_engine:
            key = f"amc_{amc_engine}_gbps"
            vals = [r[key] for r in fgr if key in r]
            if vals:
                base = 0.0 if math.isnan(amc_baseline) else amc_baseline
                return (np.mean(vals) - base) * 1e9, "bytes/s (AMC counter)"
        # 2) model-based bytes/s
        vals = [r.get("realized_bytes_ps", float("nan")) for r in bg]
        if vals and not all(math.isnan(v) for v in vals):
            return float(np.nanmean(vals)), "bytes/s (model)"
        # 3) offered ops/s
        vals = [r["offered_qps"] for r in bg
                if not math.isnan(r.get("offered_qps", float("nan")))]
        return (float(np.mean(vals)) if vals else float("nan")), "ops/s (offered)"

    points, unit = [], None
    for lvl in usable:
        x, u = xs_for(lvl)
        unit = unit or u
        if u != unit:          # never mix units in one fit
            continue
        ys = [r.get(y_key, float("nan"))
              for r in kind_rows[lvl] if r["role"] == "fg"]
        points.append((x, ys))
    return points, unit or "n/a", saturated


def band_verdict(ratio, ci):
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "NOT EVALUABLE (insufficient data for a ratio CI)"
    # Degenerate (zero-width) CI: at R=1 the bootstrap resamples a single
    # run-cluster, so every replicate is identical and lo==hi. A band verdict
    # from a zero-width interval is meaningless (it would FALSIFY for ANY point
    # estimate off the band); refuse it. Need R>=2 for a real ratio interval.
    if hi <= lo:
        return "NOT EVALUABLE (degenerate zero-width CI — need R>=2 replicate runs)"
    if BAND[0] <= lo and hi <= BAND[1]:
        return "CONSISTENT with engine-independence (CI within [2/3, 3/2])"
    if hi < BAND[0] or lo > BAND[1]:
        return "FALSIFIES engine-independence (CI wholly outside [2/3, 3/2])"
    return "INCONCLUSIVE (CI overlaps the [2/3, 3/2] band boundary)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PER_RUN_COLS = [
    "cell", "stage", "B", "kind", "level_pct", "run", "pipeline", "role",
    "scheduler", "n_queries", "n_completed", "n_used", "span_s", "anchor",
    "match_frac", "blocked_puts", "p50_resp_s", "p95_resp_s", "mean_resp_s",
    "throughput_qps", "offered_qps", "realized_qps", "offered_bytes_ps",
    "realized_bytes_ps", "gen_dur_s_med", "tok_per_s_med", "per_token_s_med",
    "ttft_s_med", "amc_cpu_gbps", "amc_gpu_gbps", "amc_ane_gbps",
    "amc_other_gbps", "amc_total_gbps", "notes",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["mlx", "cuda"], required=True)
    ap.add_argument("--results-dir", default=None,
                    help="default: evaluation/collect/results/<device>")
    ap.add_argument("--out-dir", default=None,
                    help="default: evaluation/contention/analysis/<device>")
    ap.add_argument("--plots", action="store_true",
                    help="write matplotlib figures (Agg backend)")
    ap.add_argument("--knee-tol", type=float, default=0.10,
                    help="realized-vs-offered shortfall marking saturation")
    ap.add_argument("--fg-warmup", type=int, default=None,
                    help="fg queries dropped per run (default: knobs e6 "
                         "warmup_k for A-C; 3 for D per design)")
    ap.add_argument("--bg-warmup-s", type=float, default=60.0,
                    help="co-runner warm-up window excluded from realized "
                         "rates (capped at span/4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    devname = {"mlx": "m2pro", "cuda": "gb10"}[dev]
    results_dir = Path(args.results_dir or sl.default_results_dir(dev))
    out_dir = Path(args.out_dir or (HERE / "analysis" / dev))
    out_dir.mkdir(parents=True, exist_ok=True)

    knobs = sl.load_knobs()
    wk_abc = args.fg_warmup if args.fg_warmup is not None else int(
        sl.get_knob(knobs, "e6", devname, "warmup_k", 2) or 2)
    wk_d = args.fg_warmup if args.fg_warmup is not None else 3

    cells = discover(results_dir, dev)
    expected = sorted(p.stem for p in CONFIG_DIR.glob(f"stage_*_{dev}.yml"))

    report = [f"Staged contention analysis — device={dev}",
              f"results: {results_dir}", ""]
    missing = []
    for cell in expected:
        have = sorted(cells.get(cell, {}))
        if len(have) < EXPECTED_R:
            missing.append(f"  {cell}: runs present {have or '[]'} "
                           f"(expected r1..r{EXPECTED_R})")
    if not expected:
        report.append("WARNING: no configs found under evaluation/contention/"
                      "configs — expected-cell audit skipped.")
    stray = sorted(set(cells) - set(expected))
    if stray:
        report.append("Cells present without a matching config (analyzed with "
                      "name-prefix role heuristics): " + ", ".join(stray))

    # ---- per-run pass -----------------------------------------------------
    per_run: list[dict] = []
    for cell in sorted(cells):
        meta = parse_cell_meta(cell)
        cfg = load_cell_config(cell)
        wk = wk_d if meta["stage"] == "d" else wk_abc
        for run in sorted(cells[cell]):
            try:
                per_run.extend(analyze_run(cell, meta, run, cells[cell][run],
                                           cfg, wk, args.bg_warmup_s))
            except Exception as exc:  # a corrupt run must not kill the pass
                per_run.append({"cell": cell, **meta, "run": run,
                                "pipeline": "", "role": "error",
                                "notes": f"PARSE FAILED: {exc}"})

    with open(out_dir / "staged_per_run.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_RUN_COLS, extrasaction="ignore")
        w.writeheader()
        for r in per_run:
            w.writerow({k: (sl.fmt(v) if isinstance(v, float) else v)
                        for k, v in r.items() if k in PER_RUN_COLS})

    # ---- cell-level estimates ----------------------------------------------
    by_cell = defaultdict(list)
    for r in per_run:
        if r["role"] in ("fg", "bg"):
            by_cell[r["cell"]].append(r)
    cell_rows: list[dict] = []
    for cell in sorted(by_cell):
        cell_rows.extend(agg_cell(by_cell[cell], args.seed))

    def cellrows(stage, **filt):
        sel = [r for r in per_run if r["stage"] == stage
               and r["role"] in ("fg", "bg")]
        for k, v in filt.items():
            sel = [r for r in sel if r.get(k) == v]
        return sel

    def cell_est(cell, metric):
        for r in cell_rows:
            if r["cell"] == cell and r["metric"] == metric:
                return r
        return None

    # ---- Stage A ------------------------------------------------------------
    report += ["=" * 72, "STAGE A — system view: fg degradation vs B", ""]
    a_cells = [c for c in sorted(by_cell) if parse_cell_meta(c)["stage"] == "a"]
    if a_cells:
        report.append(f"  {'B':>2} {'p50 resp (s)':>24} {'p95 resp (s)':>24} "
                      f"{'fg thr (1/s)':>22} {'bg realized (q/s)':>20}")
        for c in a_cells:
            B = parse_cell_meta(c)["B"]
            cols = []
            for met in ("fg_p50_resp_s", "fg_p95_resp_s", "fg_throughput_qps",
                        "bg_realized_qps"):
                e = cell_est(c, met)
                cols.append(f"{sl.fmt(e['estimate'])} "
                            f"[{sl.fmt(e['ci_lo'])},{sl.fmt(e['ci_hi'])}]"
                            if e else "—")
            report.append(f"  {B:>2} {cols[0]:>24} {cols[1]:>24} "
                          f"{cols[2]:>22} {cols[3]:>20}")
        report.append("  (hierarchical run-cluster 95% CIs; p95 emitted only "
                      "past the >=500-pooled-queries gate)")
    else:
        report.append("  no Stage-A cells collected yet")
    report.append("")

    # ---- Stage B ------------------------------------------------------------
    report += ["=" * 72, "STAGE B — dose-response vs offered bg intensity", ""]
    b_cells = [c for c in sorted(by_cell) if parse_cell_meta(c)["stage"] == "b"]
    if b_cells:
        report.append(f"  {'L%':>4} {'offered q/s':>12} {'realized q/s':>13} "
                      f"{'AMC gpu GB/s':>13} {'p50 resp (s)':>22}")
        base = f"stage_a_B0_{dev}"
        for c in ([base] if base in by_cell else []) + b_cells:
            meta = parse_cell_meta(c)
            lvl = 0 if c == base else meta["level_pct"]
            bgr = [r for r in by_cell[c] if r["role"] == "bg"]
            off = next((r["offered_qps"] for r in bgr
                        if not math.isnan(r.get("offered_qps", float("nan")))),
                       float("nan"))
            e_bg = cell_est(c, "bg_realized_qps")
            e_amc = cell_est(c, "amc_gpu_gbps")
            e50 = cell_est(c, "fg_p50_resp_s")
            report.append(
                f"  {lvl:>4} {sl.fmt(off):>12} "
                f"{sl.fmt(e_bg['estimate']) if e_bg else '—':>13} "
                f"{sl.fmt(e_amc['estimate']) if e_amc else '—':>13} "
                f"{(sl.fmt(e50['estimate']) + ' [' + sl.fmt(e50['ci_lo']) + ',' + sl.fmt(e50['ci_hi']) + ']') if e50 else '—':>22}")
        if not any(cell_est(c, "amc_gpu_gbps") for c in b_cells):
            report.append("  (no AMC bandwidth CSVs found for these runs — "
                          "x-axis stays in ops/s until counter traces land)")
    else:
        report.append("  no Stage-B cells collected yet")
    report.append("")

    # ---- Stage C/D slopes ----------------------------------------------------
    slope_store: dict[tuple[str, str], dict] = {}
    for stage, y_key, y_desc in (("c", "p50_resp_s", "fg median response (s)"),
                                 ("d", "tok_per_s_med", "fg decode tok/s")):
        report += ["=" * 72,
                   f"STAGE {stage.upper()} — per-engine dose-response "
                   f"(y = {y_desc})", ""]
        rows_s = cellrows(stage)
        kinds = sorted({r["kind"] for r in rows_s})
        if not kinds:
            report.append(f"  no Stage-{stage.upper()} cells collected yet\n")
            continue
        # isolation baseline for the AMC delta
        amc_base = {}
        base_rows = [r for r in cellrows("a", B=0) if r["role"] == "fg"]
        for eng in ("cpu", "gpu", "ane"):
            vals = [r[f"amc_{eng}_gbps"] for r in base_rows
                    if f"amc_{eng}_gbps" in r]
            amc_base[eng] = (sum(vals) / len(vals)) if vals else float("nan")
        for kind in kinds:
            krows = defaultdict(list)
            for r in rows_s:
                if r["kind"] == kind:
                    krows[int(r["level_pct"])].append(r)
            eng = KIND_ENGINE.get(kind)
            points, unit, saturated = knee_and_points(
                krows, y_key, args.knee_tol, eng, amc_base.get(eng, float("nan")))
            # x=0 isolation anchor: Stage C shares the RAG fg with stage_a_B0;
            # Stage D's decode fg has no collected isolation cell -> intercept
            # of the fit is the zero-dose baseline instead.
            if stage == "c":
                iso = [r.get(y_key, float("nan"))
                       for r in cellrows("a", B=0) if r["role"] == "fg"]
                if iso and not all(math.isnan(v) for v in iso):
                    points = [(0.0, iso)] + points
            res = sl.slope_boot_samples(points, seed=args.seed)
            report.append(f"  co-runner {kind} (engine {eng}), x-unit: {unit}")
            if saturated:
                report.append(f"    saturation knee: levels {saturated}% "
                              f"excluded (realized < {1-args.knee_tol:.0%} "
                              f"of offered)")
            if res is None:
                report.append("    < 2 usable ladder levels — no slope\n")
                continue
            samples, slope, intercept, nsamples, nslope = res
            s_ci = sl.ci_of(samples)
            n_ci = sl.ci_of(nsamples)
            report.append(f"    slope: {sl.fmt(slope)} per {unit} "
                          f"[{sl.fmt(s_ci[0])}, {sl.fmt(s_ci[1])}]")
            report.append(f"    normalized slope (fraction of zero-dose "
                          f"baseline per {unit}): {sl.fmt(nslope)} "
                          f"[{sl.fmt(n_ci[0])}, {sl.fmt(n_ci[1])}]")
            if s_ci[0] <= 0.0 <= s_ci[1]:
                report.append("    slope CI includes 0 — FLAT curve "
                              "(publishable per design: co-runner effectively "
                              "free over the measured dose range)")
            slope_store[(stage, kind)] = {
                "samples": nsamples, "nslope": nslope, "ci": n_ci,
                "unit": unit, "slope": slope, "slope_ci": s_ci}
            cell_rows.append({"cell": f"stage_{stage}[{kind}]", "stage": stage,
                              "B": float("nan"), "kind": kind,
                              "level_pct": float("nan"),
                              "metric": f"norm_slope[{y_key}]",
                              "estimate": nslope, "ci_lo": n_ci[0],
                              "ci_hi": n_ci[1],
                              "n_runs": sum(len(ys) for _, ys in points),
                              "unit": f"frac per {unit}", "run_values": ""})
        # pairwise ratios at matched bytes/s
        report.append("")
        byte_kinds = [k for k in kinds if (stage, k) in slope_store
                      and slope_store[(stage, k)]["unit"].startswith("bytes")]
        for i, k1 in enumerate(byte_kinds):
            for k2 in byte_kinds[i + 1:]:
                a, b = slope_store[(stage, k1)], slope_store[(stage, k2)]
                if a["unit"] != b["unit"]:
                    continue
                import numpy as np
                ratio = (a["nslope"] / b["nslope"]) if b["nslope"] else float("nan")
                rci = sl.ratio_ci(a["samples"], b["samples"])
                verdict = band_verdict(ratio, rci)
                report.append(f"  slope ratio {k1}/{k2} at matched {a['unit']}: "
                              f"{sl.fmt(ratio)} [{sl.fmt(rci[0])}, "
                              f"{sl.fmt(rci[1])}] -> {verdict}")
                cell_rows.append({"cell": f"stage_{stage}[{k1}/{k2}]",
                                  "stage": stage, "B": float("nan"),
                                  "kind": f"{k1}/{k2}",
                                  "level_pct": float("nan"),
                                  "metric": "norm_slope_ratio",
                                  "estimate": ratio, "ci_lo": rci[0],
                                  "ci_hi": rci[1], "n_runs": float("nan"),
                                  "unit": a["unit"], "run_values": verdict})
        if len(byte_kinds) < 2:
            report.append("  (fewer than two co-runners have a bytes/s axis — "
                          "H1 slope ratios need AMC counter CSVs or the "
                          "stream model; ops/s slopes are NOT compared "
                          "across engines)")
        report.append("")

    # ---- Verdicts ------------------------------------------------------------
    report += ["=" * 72, "PRE-REGISTERED HYPOTHESIS VERDICTS", ""]
    d_ratios = [r for r in cell_rows if r["metric"] == "norm_slope_ratio"
                and r["stage"] == "d"]
    c_ratios = [r for r in cell_rows if r["metric"] == "norm_slope_ratio"
                and r["stage"] == "c"]
    ratios = d_ratios or c_ratios
    d_slopes = {k: v for (s, k), v in slope_store.items() if s == "d"}
    if not slope_store:
        report.append("H1: NOT EVALUABLE — no Stage-C/D dose-response cells "
                      "collected yet.")
    elif not ratios:
        report.append("H1: NOT EVALUABLE at matched bytes/s — dose-response "
                      "slopes exist but fewer than two co-runners share a "
                      "bytes/s x-axis (need AMC bandwidth sidecars, mlx only, "
                      "or model-based traffic).")
    else:
        verdicts = [r["run_values"] for r in ratios]
        if all(v.startswith("NOT EVALUABLE") for v in verdicts):
            report.append(
                "H1: NOT EVALUABLE — every pairwise slope-ratio CI is degenerate "
                "(R<2 gives no ratio variance). At R=1 the point ratios are "
                "directional only; collect replicate runs for a bootstrap/Fieller "
                "ratio interval before issuing a band verdict.")
        elif any(v.startswith("FALSIFIES") for v in verdicts):
            report.append(
                "H1 FALSIFIED: at matched bytes/s, foreground degradation "
                "per unit co-runner traffic DEPENDS on the engine generating "
                "it — at least one pairwise slope-ratio CI lies wholly "
                "outside the pre-registered [2/3, 3/2] band. The bandwidth-"
                "not-compute claim does not survive in its engine-"
                "independent form; report per-engine laws instead.")
        elif all(v.startswith("CONSISTENT") for v in verdicts):
            report.append(
                "H1 SUPPORTED: foreground degradation tracks co-runner "
                "bytes/s approximately independently of the generating "
                "engine (all pairwise slope-ratio CIs within [2/3, 3/2]).")
        else:
            report.append(
                "H1 INCONCLUSIVE: at least one pairwise slope-ratio CI "
                "overlaps the [2/3, 3/2] band boundary; collect more runs or "
                "report the interval verbatim.")
        flat = [k for k, v in d_slopes.items()
                if v["slope_ci"][0] <= 0.0 <= v["slope_ci"][1]]
        if flat and len(flat) == len(d_slopes):
            report.append(
                "  note: ALL Stage-D slope CIs include 0 — the flat-curve "
                "outcome ('co-runners are effectively free up to the "
                "measured dose') is itself the publishable finding; the "
                "ratio band is then moot.")
    report.append("")
    ttft_any = any(not math.isnan(r.get("ttft_s_med", float("nan")))
                   for r in per_run if r["role"] == "fg")
    if not cellrows("d"):
        report.append("H2: NOT EVALUABLE — no Stage-D cells collected yet.")
    elif not ttft_any:
        report.append(
            "H2 (TTFT flat vs per-token degrading) NOT EVALUABLE from current "
            "traces: the generator stages log only whole-generate stage "
            "spans (stages/llm_mlx/inference.py:98-104, stages/llm_"
            "huggingface/inference.py:198-202 — single black-box generate() "
            "per query; no first-token event). per_token_s here = gen_dur / "
            "max_tokens and INCLUDES prefill. To evaluate H2, add a first-"
            "token/prefill sub-phase log line in the generator stage; this "
            "analyzer picks it up automatically ('<stage>::prefill' rows). "
            "Until then the phase-split negative control cannot distinguish "
            "bandwidth contention from whole-machine slowdown — per design "
            "§Counters, the counter-backed attribution carries that burden "
            "on the M2, and GB10 wording stays proxy-backed.")
    else:
        # TTFT vs per-token dose-response asymmetry, per kind
        report.append("H2 evaluation (TTFT vs per-token dose-response):")
        for kind in sorted({r["kind"] for r in cellrows("d")}):
            krows = defaultdict(list)
            for r in cellrows("d"):
                if r["kind"] == kind:
                    krows[int(r["level_pct"])].append(r)
            eng = KIND_ENGINE.get(kind)
            pt_pts, unit, _ = knee_and_points(krows, "per_token_s_med",
                                              args.knee_tol, eng, float("nan"))
            tt_pts, _, _ = knee_and_points(krows, "ttft_s_med",
                                           args.knee_tol, eng, float("nan"))
            rp = sl.slope_boot_samples(pt_pts, seed=args.seed)
            rt = sl.slope_boot_samples(tt_pts, seed=args.seed)
            if rp is None or rt is None:
                report.append(f"  {kind}: insufficient ladder for slopes")
                continue
            rci = sl.ratio_ci(rt[3], rp[3])
            line = (f"  {kind}: ttft-slope/per-token-slope (normalized, per "
                    f"{unit}) = {sl.fmt(rt[4] / rp[4]) if rp[4] else 'nan'} "
                    f"[{sl.fmt(rci[0])}, {sl.fmt(rci[1])}]")
            if math.isnan(rci[0]) or math.isnan(rci[1]) or rci[1] <= rci[0]:
                # degenerate zero-width CI (R=1): the point ratio is directional
                # only — do NOT issue a SUPPORTED/FALSIFIED verdict from it.
                line += " -> H2 NOT EVALUABLE (degenerate CI — need R>=2 runs)"
            elif rci[1] < 1.0 and rp[2] and rp[1] > 0:
                line += " -> H2 SUPPORTED (prefill degrades less than decode)"
            elif rci[0] >= 1.0:
                line += (" -> H2 FALSIFIED (prefill degrades at least as much "
                         "as decode — pattern consistent with thermal/"
                         "scheduling, not bandwidth)")
            else:
                line += " -> H2 INCONCLUSIVE"
            report.append(line)
    report.append("")

    # ---- missing cells --------------------------------------------------------
    report += ["=" * 72, "COLLECTION STATUS (mid-flight tolerant)", ""]
    if missing:
        report.append(f"Missing/partial cells ({len(missing)} of "
                      f"{len(expected)} expected):")
        report += missing
    else:
        report.append("All expected cells present at R=5." if expected else
                      "No expected-cell manifest (configs absent).")
    report.append("")

    # ---- write outputs ---------------------------------------------------------
    est_cols = ["cell", "stage", "B", "kind", "level_pct", "metric",
                "estimate", "ci_lo", "ci_hi", "n_runs", "unit", "run_values"]
    with open(out_dir / "staged_cell_estimates.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=est_cols, extrasaction="ignore")
        w.writeheader()
        for r in cell_rows:
            w.writerow({k: (sl.fmt(v) if isinstance(v, float) else v)
                        for k, v in r.items()})
    (out_dir / "staged_report.txt").write_text("\n".join(report) + "\n",
                                               encoding="utf-8")

    if args.plots:
        _plots(per_run, cell_rows, out_dir, dev)

    print(f"[analyze_staged] {len(cells)} cells, "
          f"{sum(len(r) for r in cells.values())} runs analyzed -> {out_dir}")
    print(f"[analyze_staged] missing/partial: {len(missing)}")
    for line in report:
        if line.startswith(("H1", "H2")):
            print(f"[analyze_staged] {line.splitlines()[0][:100]}")
    return 0


def _plots(per_run, cell_rows, out_dir: Path, dev: str) -> None:
    """Optional headless figures (one per stage with data)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def est(cell, metric):
        for r in cell_rows:
            if r["cell"] == cell and r["metric"] == metric:
                return r
        return None

    # Stage A: p50/p95 vs B
    a = sorted({r["cell"] for r in per_run if r["stage"] == "a"})
    if a:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for met, mark in (("fg_p50_resp_s", "o"), ("fg_p95_resp_s", "s")):
            xs, ys, lo, hi = [], [], [], []
            for c in a:
                e = est(c, met)
                if e:
                    xs.append(parse_cell_meta(c)["B"])
                    ys.append(e["estimate"])
                    lo.append(e["estimate"] - e["ci_lo"])
                    hi.append(e["ci_hi"] - e["estimate"])
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], marker=mark, capsize=3,
                            label=met.replace("fg_", "").replace("_s", ""))
        ax.set_xlabel("background indexers B")
        ax.set_ylabel("fg response (s)")
        ax.set_xticks([0, 1, 2])
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"stage_a_{dev}.png", dpi=150)
        plt.close(fig)

    # Stage C/D: dose-response per kind
    for stage, ykey, ylab in (("c", "p50_resp_s", "fg p50 response (s)"),
                              ("d", "tok_per_s_med", "fg decode tok/s")):
        rows = [r for r in per_run if r["stage"] == stage and r["role"] == "fg"
                and not math.isnan(r.get(ykey, float("nan")))]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for kind in sorted({r["kind"] for r in rows}):
            pts = defaultdict(list)
            for r in rows:
                if r["kind"] == kind:
                    pts[r["level_pct"]].append(r[ykey])
            xs = sorted(pts)
            ys = [sum(pts[x]) / len(pts[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=kind)
        ax.set_xlabel("offered co-runner intensity (% R_max)")
        ax.set_ylabel(ylab)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"stage_{stage}_{dev}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

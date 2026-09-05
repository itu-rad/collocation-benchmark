#!/usr/bin/env python3
"""Generate the staged contention experiment's config family (Stages A-D).

Design of record: CONTENTION_EXPERIMENTS_REDESIGN.md §0.3 (signed off
2026-07-14). The section's rhetorical core is that each stage transition
changes EXACTLY ONE config element — this generator both produces the configs
and PROVES the discipline by emitting consecutive unified diffs (DIFFS.md):

  Stage A  fg RAG-serve + B in {0,1,2} background indexers (saturating)
  Stage B  = A(B=1) with ONLY the background loadgen block changed
             (saturating -> fixed-interval at {25,50,75,100}% of R_max)
  Stage C  = B with ONLY the background stage list changed
             (indexer -> single-resource co-runner: cpu stream / gpu encode
              [/ ane encode when unblocked])
  Stage D  = C with ONLY the foreground stage list changed
             (RAG-serve -> bare decode; prefill/decode split in analysis)

Foreground pipelines are taken verbatim from the committed, knob-locked
configs (pipeline_configs/rag_serve_plain*.yml, pipeline_configs/pilots/
decode_9b_*.yml) — single source of truth. Intensity levels come from
knobs.yml (R-INTENSITY); missing pilots leave placeholders and a warning.

    python evaluation/contention/generate_stage_configs.py --device mlx
"""

from __future__ import annotations

import argparse
import copy
import difflib
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "pilots"))
import pilot_lib as pl  # noqa: E402

OUT_DIR = HERE / "configs"

# Foreground arrival rate, PER DEVICE. The source configs carry 0.2243/s for
# both, which was derived as rho ~= 0.6 of the UNCONTENDED service time. That is
# the wrong reference for this section: it leaves no headroom for the contention
# the experiment exists to create, so the queue goes unstable and the cell never
# converges.
#
#   device  service time (uncontended, measured in 5.1)  rho at 0.2243/s
#   gb10    0.70 s                                        0.16   <- ample headroom
#   m3pro   2.70 s                                        0.61   <- saturates under load
#
# rho is now set from the uncontended service time so that it stays near 0.6 at
# the WORST expected contention (~2x), i.e. rho_uncontended <= 0.3. gb10 already
# satisfies that and is left alone; m3pro drops to 0.11/s. max_queries is trimmed
# with it so a cell stays around 12 minutes.
#
# TODO: this belongs in the pilot knob registry (R-LAMBDA-BELOW-SAT) rather than
# here, keyed on a measured per-device service time.
# Round numbers: the CHOSEN knob is round, and rho is what falls out of it
# against the measured service time. The inherited 0.2243/s was precise-looking
# without being principled -- it implied rho = 0.61 on m3pro, which is not a
# number anyone picked.
#
#   device  service time (5.1)  lambda   -> rho
#   m3pro   2.70 s              0.1/s       0.27
#   gb10    0.70 s              0.2/s       0.14
FG_RATE = {"mlx": 0.1, "cuda": 0.2}
FG_MAX_QUERIES = {"mlx": 80, "cuda": 100}

FG_RAGSERVE = {"mlx": "pipeline_configs/rag_serve_plain.yml",
               "cuda": "pipeline_configs/rag_serve_plain_cuda.yml"}
FG_DECODE = {"mlx": "pipeline_configs/pilots/decode_9b_mlx.yml",
             "cuda": "pipeline_configs/pilots/decode_9b_cuda.yml"}
GPU_DEVICE = {"mlx": "mps", "cuda": "cuda"}
DEVNAME = {"mlx": "m2pro", "cuda": "gb10"}

# Background docs-per-cell window (bounded work; wraps via deterministic ids)
BG_DOCS_PER_QUERY = 32
BG_MAX_QUERIES = 100          # 3200 docs/cell
INTENSITY_LEVELS = (25, 50, 75, 100)

# --- Extended dose ladder (decision 2, --extended) --------------------------
# The base ladder measures the foreground either idle (Stage A B=0) or against a
# single saturating co-runner — two points, not a dose-response curve. A mock
# reviewer flagged that a single saturating measurement cannot separate a truly
# engine-independent slowdown from a coincidence at one operating point. The
# extended ladder adds, WITHOUT touching any base filename:
#   * deeper background fan-out (B in {3,4}) at Stage A;
#   * a foreground held at a fixed fraction (0.7, 0.8) of its own R_max, so the
#     fg sits in its contention-sensitive region instead of idle-or-saturated;
#   * stacked STREAM co-runners (2x, 3x concurrent MemoryStream pipelines) to
#     push the memory-bandwidth dose past what one stream delivers.
EXTENDED_B_LEVELS = (3, 4)
FG_LOAD_FRACTIONS = (0.7, 0.8)
STREAM_STACK_COUNTS = (2, 3)


def _apply_fg_rate(pipe: dict, device: str) -> dict:
    """Set the foreground's arrival rate and query budget for this device."""
    lg = pipe.get("loadgen") or {}
    cfg = lg.get("config") or {}
    if "rate" in cfg and device in FG_RATE:
        cfg["rate"] = FG_RATE[device]
        lg["max_queries"] = FG_MAX_QUERIES[device]
    return pipe


def _load_pipeline(rel_path: str) -> dict:
    doc = yaml.safe_load((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    return copy.deepcopy(doc["pipelines"][0])


def _bg_indexer_pipeline(shard_i: int, shard_n: int, device: str) -> dict:
    """Background indexing pipeline: chunk -> embed (GPU) -> chroma insert."""
    return {
        "name": f"BG indexer {shard_i}",
        "inputs": [0],
        "outputs": [2],
        "dataset_stage_id": 0,
        "loadgen": {
            "component": "loadgen.SaturatingOfflineScheduler",
            "queue_depth": BG_MAX_QUERIES,        # R-QDEPTH: >= total samples
            "max_queries": BG_MAX_QUERIES,
            "timeout": 3600,
            "config": {"rate": 0},
        },
        "stages": [
            {"name": f"Corpus chunk loader (shard {shard_i}/{shard_n})",
             "id": 0, "outputs": [1],
             "component": "stages.rag_indexing.CorpusChunkLoader",
             "config": {
                 "dataset": {"name": "rag-datasets/rag-mini-wikipedia",
                             "subset": "text-corpus", "split": "passages",
                             "text_column": "passage"},
                 "docs_per_query": BG_DOCS_PER_QUERY,
                 "shard": {"index": shard_i, "count": shard_n},
             }},
            {"name": "Embed passages", "id": 1, "outputs": [2],
             "component": "stages.rag_indexing.EmbedStage",
             "config": {"model": {"name": "sentence-transformers/all-MiniLM-L6-v2"},
                        "device": GPU_DEVICE[device], "max_length": 256}},
            {"name": "Chroma indexer", "id": 2,
             "component": "stages.rag_indexing.ChromaIndexer",
             # separate-stores control: fg never reads bg_index_shard*
             "config": {"collection_name": f"bg_index_shard{shard_i}"}},
        ],
    }


def _bg_corunner_stages(kind: str, device: str) -> list[dict]:
    """Stage-C single-resource co-runner stage lists (same shape: 1 stage)."""
    if kind == "stream":
        return [{"name": "CPU memory stream", "id": 0,
                 "component": "stages.evaluation.MemoryStream",
                 "config": {"size_mb": 256, "passes": 4}}]
    if kind == "clipgpu":
        return [{"name": "VQA dataloader", "id": 0, "outputs": [1],
                 "component": "stages.multimodal_vqa.VQADataLoader",
                 "config": {"dataset": {"name": "Multimodal-Fatima/OK-VQA_train",
                                        "split": "train", "image_column": "image",
                                        "question_column": "question",
                                        "answers_column": "answers",
                                        "max_samples": 100},
                            "batch_size": 1}},
                {"name": "CLIP vision encoder", "id": 1,
                 "component": "stages.multimodal_vqa.CLIPVisionEncoder",
                 "config": {"model": {"name": "openai/clip-vit-large-patch14"},
                            "device": GPU_DEVICE[device]}}]
    if kind == "clipane":
        return [{"name": "VQA dataloader", "id": 0, "outputs": [1],
                 "component": "stages.multimodal_vqa.VQADataLoader",
                 "config": {"dataset": {"name": "Multimodal-Fatima/OK-VQA_train",
                                        "split": "train", "image_column": "image",
                                        "question_column": "question",
                                        "answers_column": "answers",
                                        "max_samples": 100},
                            "batch_size": 1}},
                {"name": "CLIP vision encoder (CoreML)", "id": 1,
                 "component": "stages.multimodal_vqa.CLIPVisionEncoderCoreML",
                 "config": {"model": {"coreml_path": "tmp/clip_vit_l14_vision.mlpackage",
                                      "hf_name": "openai/clip-vit-large-patch14"},
                            "device": "ane"}}]
    raise ValueError(kind)


def _corunner_dataset_stage_meta(kind: str) -> dict:
    """dataset_stage_id + outputs for each co-runner shape."""
    if kind == "stream":
        return {"dataset_stage_id": 0, "inputs": [0], "outputs": [0]}
    return {"dataset_stage_id": 0, "inputs": [0], "outputs": [1]}


# Section 5.2 IS the counters, so every config it generates declares listeners.
# radt reads them from this key; a config without it runs with none, silently.
LISTENERS = {"mlx": ["macmon"], "cuda": ["dcgmi", "top"]}

# ---- Matched-bytes intensity (the collocation axis) -------------------------
# The L25..L100 ladder is a fraction of each co-runner's OWN saturating rate, so
# "L50" means 8.4 q/s of memory-stream against 9.2 q/s of GPU encode against
# 10.0 q/s of ANE encode. Measured, that is 51 / 32 / 13 GB/s -- a 4x spread in
# pressure on the very resource this section says they share. The types cells are
# therefore matched on BYTES/S instead, with per-co-runner rates derived from
# measured bytes/query (calibrate_bytes.py, run on the M2 Pro because the M3 Pro
# does not populate the AMC counters).
#
# The target is bounded by the ANE, which tops out around 20 GB/s -- that ceiling
# is what caps a bytes-matched comparison, and is worth stating in the paper. 12
# GB/s keeps every co-runner well inside its solo capacity (13% / 27% / 59%), so
# the offered rate is actually delivered and the match holds. That matters here
# because delivered bytes CANNOT be verified after the fact on m3pro.
MATCHED_GBPS = 12.0
# The dose-response ladder, in the same units and equally round. Capped by the
# ANE ceiling (~20 GB/s), and it includes the matched level so the dose curve and
# the types cells share a point.
DOSE_GBPS = (4.0, 8.0, 12.0, 16.0)
BYTES_CALIBRATION = HERE / "bytes_calibration.json"


def _bytes_interval(kind: str, gbps: float) -> float | None:
    """Seconds between queries for `kind` to deliver `gbps`, or None.

    The interval is not round, and should not be: it is derived from a MEASURED
    bytes/query. The round number is the target it delivers.
    """
    import json
    try:
        cal = json.loads(BYTES_CALIBRATION.read_text())
    except FileNotFoundError:
        return None
    entry = (cal.get("corunners") or {}).get(kind)
    if not entry or not entry.get("bytes_per_query"):
        return None
    qps = gbps * 1e9 / entry["bytes_per_query"]
    return round(1.0 / qps, 5)

# The background must outlast the foreground. Its own query budget used to end
# the cell, but it is sized independently of the foreground's, and at L50 the
# co-runner ran ~119s against a ~446s foreground -- so most of the cell would be
# measured UNCONTENDED and every degradation number quietly diluted. Size it from
# the foreground's own loadgen instead; overshooting is free, the run ends with
# the foreground.
BG_MARGIN = 1.5


def _pipeline_seconds(pipe: dict) -> float:
    """Wall-clock a pipeline's loadgen is expected to occupy."""
    lg = pipe.get("loadgen") or {}
    n = float(lg.get("max_queries") or 0)
    cfg = lg.get("config") or {}
    rate = float(cfg.get("rate") or 0)
    interval = float(cfg.get("interval") or 0)
    if rate > 0:
        return n / rate
    if interval > 0:
        return n * interval
    return 0.0


def _size_backgrounds(doc: dict) -> None:
    """Give every background pipeline enough queries to cover the foreground."""
    pipes = doc.get("pipelines") or []
    if len(pipes) < 2:
        return
    fg_s = _pipeline_seconds(pipes[0])
    if fg_s <= 0:
        return
    for bg in pipes[1:]:
        lg = bg.get("loadgen") or {}
        cfg = lg.get("config") or {}
        interval = float(cfg.get("interval") or 0)
        if interval <= 0:
            continue
        # Query budget covers the foreground with margin, so the background never
        # runs dry mid-cell...
        need = fg_s * BG_MARGIN
        lg["max_queries"] = int(-(-need // interval))
        # ...and the TIMEOUT is what actually ends it, tied to the foreground
        # rather than to the background's own budget. Under contention the
        # background does not achieve its nominal rate, so a budget-derived
        # timeout does not bound the cell at all: one MPS cell ran 92 minutes
        # instead of 12, with the foreground long finished and radt waiting on
        # the background. The foreground defines the measurement window; the
        # background is stopped just past it.
        lg["timeout"] = int(fg_s * 1.2)
        # A deep queue defeats the timeout. The generator paces at `interval`,
        # but under contention the co-runner sustains far less, so a 1000-deep
        # queue saturates and the pipeline is still draining that backlog long
        # after generation has stopped -- which is how a bounded cell still
        # outlived its foreground by 12 minutes and counting. A background is
        # meant to apply steady pressure, not a burst with a long tail, so keep
        # its backlog shallow and the timeout means what it says.
        lg["queue_depth"] = 16


def _dump(doc: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = name.rsplit("_", 1)[-1].removesuffix(".yml").removesuffix("_mps")
    if dev in LISTENERS:
        doc.setdefault("listeners", list(LISTENERS[dev]))
    _size_backgrounds(doc)
    path = OUT_DIR / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=100),
                    encoding="utf-8")
    return path


def _stream_stack_pipelines(count: int, device: str, rate: float | None) -> list[dict]:
    """`count` concurrent MemoryStream co-runner pipelines (stacked dose)."""
    pipes = []
    for j in range(count):
        bg = {"name": f"BG stream {j}/{count}"}
        bg.update(_corunner_dataset_stage_meta("stream"))
        bg["loadgen"] = {
            "component": "loadgen.MultiStreamScheduler",
            "queue_depth": 1000,
            "max_queries": 1000,
            "timeout": 3600,
            "config": {"interval": round(1.0 / rate, 5) if rate else None},
        }
        bg["stages"] = _bg_corunner_stages("stream", device)
        pipes.append(bg)
    return pipes


def _throttle_fg(fg_doc: dict, rate: float | None) -> dict:
    """Return a copy of a foreground pipeline held at a fixed arrival rate
    (fixed-interval MultiStream) instead of its native saturating loadgen."""
    doc = copy.deepcopy(fg_doc)
    lg = dict(doc.get("loadgen", {}))
    lg["component"] = "loadgen.MultiStreamScheduler"
    cfg = dict(lg.get("config", {}))
    cfg["interval"] = round(1.0 / rate, 5) if rate else None
    lg["config"] = cfg
    doc["loadgen"] = lg
    return doc


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["mlx", "cuda"], required=True)
    ap.add_argument("--include-ane", action="store_true",
                    help="emit the Stage-C ANE co-runner variant (blocked on "
                         "the CoreML hang fix)")
    ap.add_argument("--extended", action="store_true",
                    help="ALSO emit the extended dose ladder (decision 2): B in "
                         "{3,4}, fg-throttled arms, stacked STREAM co-runners. "
                         "Uses distinct filenames — never overwrites base configs.")
    ap.add_argument("--out-dir", default=None,
                    help="override the config output directory (default "
                         "evaluation/contention/configs; use a scratch dir to "
                         "dry-run generation without touching the live configs).")
    args = ap.parse_args()
    dev = args.device
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    knobs = pl.load_knobs()
    devname = DEVNAME[dev]

    def knob(exp, name, default=None):
        return pl.get_knob(knobs, exp, devname, name, default)

    # Stage-C/B intensity: fixed-interval rates per co-runner from knobs
    # (R-INTENSITY); the bg indexer's own R_max needs the e6p_bg_index pilot.
    warn = []
    files: list[tuple[str, Path]] = []
    fg = _apply_fg_rate(_load_pipeline(FG_RAGSERVE[dev]), dev)

    # ---- Stage A: B in {0,1,2} ------------------------------------------
    for B in (0, 1, 2):
        doc = {"name": f"stage_a_B{B}_{dev}",
               "pipelines": [copy.deepcopy(fg)]
               + [_bg_indexer_pipeline(i, max(B, 1), dev) for i in range(B)]}
        files.append((f"A B={B}", _dump(doc, f"stage_a_B{B}_{dev}.yml")))

    # ---- Stage B: = A(B=1), ONLY the bg loadgen block changes ------------
    bg_rmax = knob("e6", "bg_index_rmax_qps")
    if bg_rmax is None:
        warn.append("bg indexer R_max pilot missing (e6p_bg_index_rmax) — "
                    "Stage B rates are PLACEHOLDERS (rate: null)")
    for lvl in INTENSITY_LEVELS:
        doc = {"name": f"stage_b_L{lvl}_{dev}",
               "pipelines": [copy.deepcopy(fg), _bg_indexer_pipeline(0, 1, dev)]}
        rate = round(bg_rmax * lvl / 100, 4) if bg_rmax else None
        doc["pipelines"][1]["loadgen"] = {
            "component": "loadgen.MultiStreamScheduler",
            "queue_depth": BG_MAX_QUERIES,
            "max_queries": BG_MAX_QUERIES,
            "timeout": 3600,
            "config": {"interval": round(1.0 / rate, 4) if rate else None},
        }
        files.append((f"B L={lvl}%", _dump(doc, f"stage_b_L{lvl}_{dev}.yml")))

    # ---- Stage C: = B, ONLY the bg stage list changes --------------------
    corunners = ["stream", "clipgpu"] + (["clipane"] if args.include_ane else [])
    for kind in corunners:
        rmax_knob = {"stream": "corunner_c3_levels",
                     "clipgpu": "corunner_c1_gpu_levels",
                     "clipane": "corunner_c2_ane_levels"}[kind]
        levels = knob("e3", rmax_knob)
        if levels is None:
            warn.append(f"co-runner {kind}: no R-INTENSITY levels in knobs — "
                        "placeholder intervals")
        for i, lvl in enumerate(INTENSITY_LEVELS):
            doc = {"name": f"stage_c_{kind}_L{lvl}_{dev}",
                   "pipelines": [copy.deepcopy(fg), None]}
            bg = {"name": f"BG co-runner {kind}"}
            bg.update(_corunner_dataset_stage_meta(kind))
            rate = levels[i] if levels else None
            bg["loadgen"] = {
                "component": "loadgen.MultiStreamScheduler",
                "queue_depth": 1000,
                "max_queries": 1000,
                "timeout": 3600,
                "config": {"interval": round(1.0 / rate, 5) if rate else None},
            }
            bg["stages"] = _bg_corunner_stages(kind, dev)
            doc["pipelines"][1] = bg
            files.append((f"C {kind} L={lvl}%",
                          _dump(doc, f"stage_c_{kind}_L{lvl}_{dev}.yml")))
            # The MPS twin: the SAME config, differing in one key. radt launches
            # each pipeline as its own process, so MPS has two processes to
            # partition; `collocation` is what tells radt to bring the daemon up.
            # Time-sliced vs MPS is then a one-line diff, which is the claim.
            if lvl == INTENSITY_LEVELS[0] and dev == "mlx":
                # Matched level for every co-runner; the dose ladder for the
                # memory-stream antagonist that carries the dose-response.
                targets = set([MATCHED_GBPS])
                if kind == "stream":
                    targets |= set(DOSE_GBPS)
                for gbps in sorted(targets):
                    iv = _bytes_interval(kind, gbps)
                    if not iv:
                        continue
                    b_doc = copy.deepcopy(doc)
                    # "B" is already taken by stage_a_B<n> (background COUNT); this is bandwidth.
                    tag = f"GB{int(gbps)}"
                    b_doc["name"] = f"stage_c_{kind}_{tag}_{dev}"
                    b_doc["pipelines"][1]["loadgen"]["config"]["interval"] = iv
                    files.append((f"C {kind} {tag} ({gbps:g} GB/s)",
                                  _dump(b_doc, f"stage_c_{kind}_{tag}_{dev}.yml")))
            if dev == "cuda" and kind == "clipgpu":
                mps_doc = copy.deepcopy(doc)
                mps_doc["name"] = f"stage_c_{kind}_L{lvl}_{dev}_mps"
                mps_doc["collocation"] = "mps"
                files.append((f"C {kind} L={lvl}% (mps)",
                              _dump(mps_doc, f"stage_c_{kind}_L{lvl}_{dev}_mps.yml")))

    # ---- Stage D: = C, ONLY the fg pipeline changes ----------------------
    fg_decode = _load_pipeline(FG_DECODE[dev])
    for kind in corunners:
        for lvl in INTENSITY_LEVELS:
            src = OUT_DIR / f"stage_c_{kind}_L{lvl}_{dev}.yml"
            doc = yaml.safe_load(src.read_text(encoding="utf-8"))
            doc["name"] = f"stage_d_{kind}_L{lvl}_{dev}"
            doc["pipelines"][0] = copy.deepcopy(fg_decode)
            files.append((f"D {kind} L={lvl}%",
                          _dump(doc, f"stage_d_{kind}_L{lvl}_{dev}.yml")))

    # ---- Extended dose ladder (decision 2, opt-in) -----------------------
    if args.extended:
        # (E1) deeper Stage-A background fan-out
        for B in EXTENDED_B_LEVELS:
            doc = {"name": f"stage_a_B{B}_{dev}",
                   "pipelines": [copy.deepcopy(fg)]
                   + [_bg_indexer_pipeline(i, B, dev) for i in range(B)]}
            files.append((f"A B={B} (ext)", _dump(doc, f"stage_a_B{B}_{dev}.yml")))

        # (E2) foreground held at a fixed fraction of its R_max, against the
        # max-intensity stream co-runner (the contention-sensitive operating
        # point). fg R_max comes from the rag-serve pilot (e5/R-FGMAX).
        fg_rmax = knob("e5", "ragserve_fg_rmax_qps")
        if fg_rmax is None:
            warn.append("fg R_max pilot missing (e5p_ragserve_fgmax) — fg-throttle "
                        "arms are PLACEHOLDERS (interval: null)")
        stream_levels = knob("e3", "corunner_c3_levels")
        stream_top = stream_levels[-1] if stream_levels else None
        for frac in FG_LOAD_FRACTIONS:
            fg_rate = round(fg_rmax * frac, 4) if fg_rmax else None
            pct = int(round(frac * 100))
            doc = {"name": f"stage_c_stream_L100_fg{pct}_{dev}",
                   "pipelines": [_throttle_fg(fg, fg_rate)]}
            bg = {"name": "BG co-runner stream"}
            bg.update(_corunner_dataset_stage_meta("stream"))
            bg["loadgen"] = {
                "component": "loadgen.MultiStreamScheduler",
                "queue_depth": 1000, "max_queries": 1000, "timeout": 3600,
                "config": {"interval": round(1.0 / stream_top, 5) if stream_top else None},
            }
            bg["stages"] = _bg_corunner_stages("stream", dev)
            doc["pipelines"].append(bg)
            files.append((f"C stream fg={pct}% (ext)",
                          _dump(doc, f"stage_c_stream_L100_fg{pct}_{dev}.yml")))

        # (E3) stacked STREAM co-runners at max intensity (bandwidth dose > 1x)
        for cnt in STREAM_STACK_COUNTS:
            doc = {"name": f"stage_c_streamx{cnt}_L100_{dev}",
                   "pipelines": [copy.deepcopy(fg)]
                   + _stream_stack_pipelines(cnt, dev, stream_top)}
            files.append((f"C stream x{cnt} (ext)",
                          _dump(doc, f"stage_c_streamx{cnt}_L100_{dev}.yml")))

    # ---- DIFFS.md: prove the single-diff discipline -----------------------
    pairs = [(f"stage_a_B1_{dev}.yml", f"stage_b_L100_{dev}.yml",
              "A->B: only the background loadgen block"),
             (f"stage_b_L100_{dev}.yml", f"stage_c_stream_L100_{dev}.yml",
              "B->C: only the background stage list (+its loadgen units)"),
             (f"stage_c_stream_L100_{dev}.yml", f"stage_d_stream_L100_{dev}.yml",
              "C->D: only the foreground pipeline")]
    out = ["# Stage-transition diffs (single-diff discipline)",
           "", "Generated by generate_stage_configs.py — the paper shows these.", ""]
    for a, b, title in pairs:
        out += [f"## {title}", "```diff"]
        out += list(difflib.unified_diff(
            (OUT_DIR / a).read_text().splitlines(),
            (OUT_DIR / b).read_text().splitlines(),
            fromfile=a, tofile=b, lineterm=""))
        out += ["```", ""]
    (OUT_DIR / "DIFFS.md").write_text("\n".join(out), encoding="utf-8")

    print(f"[gen] wrote {len(files)} configs + DIFFS.md to {OUT_DIR}")
    for w in warn:
        print(f"[gen] WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

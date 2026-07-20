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


def _dump(doc: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    fg = _load_pipeline(FG_RAGSERVE[dev])

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

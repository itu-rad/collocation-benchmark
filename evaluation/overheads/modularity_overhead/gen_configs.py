#!/usr/bin/env python3
"""Generate the static per-(cell x device) Choreo configs for E2 (modularity).

One fully-explicit YAML per sweep cell per device — no runtime config patching
(the execution model in EXPERIMENTS.md: static configs + a bash loop). Reads the
cell manifest (configs/scale_sweep.yml) and the pipeline template
(configs/torchvision_training.yml), writes configs/generated/.

The single source of truth for a cell is its (model, weights, batch) triple: the
bare-monolith arm (baseline_finetune.py) is given the same triple on its command
line, so both arms run the IDENTICAL workload and the only difference is the
framework wrapper.

`num_workers` is forced to 0 — a control (not a tuned knob) that removes the
concurrent-prefetch data path so the per-step metric isolates the wrapper, and it
is applied identically in both arms.

    python gen_configs.py [--devices mps cuda] [--out-dir configs/generated]
"""

import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "configs", "torchvision_training.yml")
CELLS = os.path.join(HERE, "configs", "scale_sweep.yml")
NUM_WORKERS = 0

# Two different things were both called "device". The torch device string
# ("mps"/"cuda") goes INSIDE the config; the filename and run label carry the
# MACHINE, so an M2 Pro and the M3 Pro that replaces it stay distinguishable.
# evaluation/pilots/derive_knobs.py already uses this convention.
DEVICE_NAME = {"mps": "m2pro", "cuda": "gb10"}


def cell_name(cell):
    """Stable per-cell label: m<tag>_b<batch> (matches the CSV suffix scheme)."""
    return f"m{cell.get('tag', 'canonical')}_b{cell['batch']}"


def build(cell, device, template):
    cfg = yaml.safe_load(open(template, "r", encoding="utf-8"))
    # No listeners: the overhead cells run via `main.py -p 0` with RADT_PRESENT
    # unset, so nothing spawns anyway — drop them so the config states the truth.
    cfg["listeners"] = []
    cfg["name"] = f"E2 modularity {cell['model']} batch {cell['batch']} ({device})"
    pipe = cfg["pipelines"][0]
    pipe["loadgen"]["max_queries"] = int(cell.get("max_batches", 300))
    for stage in pipe["stages"]:
        comp = stage.get("component", "")
        # Both Choreo arms are logs-off. The metric of record is the step
        # PERIOD, taken from the pipeline-level rows that pipeline.py emits
        # unconditionally, and the per-stage breakdown comes from spans.
        # Leaving the per-stage CSV rows on would put a synchronous
        # write+flush inside the measured interval on one side only.
        stage["disable_logs"] = True
        if "TorchVisionClassification" in comp:
            stage["config"]["device"] = device
            stage["config"]["model"]["component"] = f"torchvision.models.{cell['model']}"
        if "TorchVisionDataLoader" in comp:
            stage["config"]["num_workers"] = NUM_WORKERS
            stage["config"]["batch_size"] = cell["batch"]
            # dataset.weights drives the preprocessing transform (input
            # resolution), so both arms feed the model the same shape.
            stage["config"]["dataset"]["weights"] = cell["weights"]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["mps", "cuda"])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs", "generated"))
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--template", default=TEMPLATE)
    args = ap.parse_args()

    cells = yaml.safe_load(open(args.cells, "r", encoding="utf-8"))["cells"]
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for device in args.devices:
        for cell in cells:
            cfg = build(cell, device, args.template)
            machine = DEVICE_NAME.get(device, device)
            path = os.path.join(args.out_dir,
                                f"mod_{cell_name(cell)}_{machine}.yml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"wrote {os.path.relpath(path, HERE)}")
            n += 1
    print(f"\n{n} configs ({len(cells)} cells x {len(args.devices)} devices), "
          f"num_workers={NUM_WORKERS}")


if __name__ == "__main__":
    main()

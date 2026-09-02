#!/usr/bin/env python3
"""Split the fused contention configs into per-process foreground and background configs.

WHY THIS EXISTS. The staged configs (stage_a..stage_d) declare the foreground
and the background as two pipelines inside ONE config, which the framework runs
as two pipelines in ONE process. That is fine for the CPU and same-engine cells,
but it makes two of section 5.2's claims unmeasurable:

  * MPS and MIG partition GPU work BETWEEN PROCESSES. With both pipelines in one
    process there is nothing for either mechanism to partition, so the gb10
    collocation axis (time-sliced / MPS / CPU) collapses to a single point.
  * The section promises every number is attributed to the pipeline that caused
    it -- each with its own run, its own listeners, its own spans. One process is
    one run, so the attribution has to be reconstructed rather than measured.

Splitting them restores both: the background becomes its own process with its
own radt run, and the collocation mechanism has two processes to separate.

The pipelines are already self-contained (each carries its own loadgen and its
own dataset_stage_id), so this is a mechanical split, not a redesign -- the
stage lists are copied verbatim from the committed staged configs.

    python evaluation/contention/gen_collocation_configs.py --device mlx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG_DIR = HERE / "configs"

# Background pipelines are named "BG ..." by generate_stage_configs.py. Keying on
# that rather than on "not the foreground" matters for stage_d, whose foreground is
# a bare decode pipeline rather than the RAG-serve one.
BG_PREFIX = "BG "


def _dump(path: Path, doc: dict) -> None:
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))


def split(device: str) -> int:
    fused = sorted(CFG_DIR.glob(f"stage_[cd]_*_{device}.yml"))
    if not fused:
        print(f"gen_collocation_configs: no stage_c/stage_d configs for {device}", file=sys.stderr)
        return 1

    # Foreground: stage_a_B0 is already the foreground alone (B=0 background).
    b0 = CFG_DIR / f"stage_a_B0_{device}.yml"
    if not b0.exists():
        print(f"gen_collocation_configs: missing {b0.name}", file=sys.stderr)
        return 1
    fg = yaml.safe_load(b0.read_text())
    fg_name = f"fg_ragserve_{device}"
    fg["name"] = fg_name
    _dump(CFG_DIR / f"{fg_name}.yml", fg)
    written = [f"{fg_name}.yml"]

    for src in fused:
        doc = yaml.safe_load(src.read_text())
        bgs = [p for p in doc["pipelines"] if p.get("name", "").startswith(BG_PREFIX)]
        if len(bgs) != 1:
            print(f"gen_collocation_configs: {src.name} has {len(bgs)} background "
                  f"pipeline(s), expected 1 -- skipped", file=sys.stderr)
            continue
        # stage_c_stream_L50_mlx -> bg_stream_L50_mlx
        m = re.match(rf"stage_[cd]_(.+?)_(L\d+)_{device}$", src.stem)
        if not m:
            print(f"gen_collocation_configs: cannot parse {src.stem} -- skipped", file=sys.stderr)
            continue
        kind, level = m.groups()
        out_name = f"bg_{kind}_{level}_{device}"
        out = CFG_DIR / f"{out_name}.yml"
        if out.exists():          # stage_c and stage_d share a background
            continue
        _dump(out, {"name": out_name, "pipelines": bgs})
        written.append(out.name)

    print(f"gen_collocation_configs [{device}]: wrote {len(written)} config(s)")
    for w in written:
        print(f"  {w}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", choices=["mlx", "cuda"], required=True)
    return split(ap.parse_args().device)


if __name__ == "__main__":
    raise SystemExit(main())

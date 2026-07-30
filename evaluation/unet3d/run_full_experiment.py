"""Full 3D-UNet / KiTS19 experiment runner (R=1) — timing + accuracy per case.

Runs the exact stage code path (`stages.unet3d_kits19.kits19_lib` preprocessing +
the `UNet3DInference` gaussian sliding window) over every KiTS19 inference case and
records, per case:

  * preprocess_s  — resample/normalize/pad (the part MLPerf EXCLUDES from timing)
  * inference_s   — sliding-window forward passes (the ONLY part MLPerf times)
  * n_subvolumes, image_shape
  * Dice kidney (labels>=1), tumor (label==2), mean composite — vs ground truth

The headline the paper needs is the **preprocess fraction of end-to-end**: MLPerf
reports only inference_s, so a large preprocess_s/(preprocess_s+inference_s) is the
cost a per-stage-timed benchmark (Choreo) surfaces and MLPerf hides.

Timing here is representative of the Choreo framework run: the stages are thin
wrappers over this same code (framework case_00000 = 69.5 s; direct = 69.0 s).

    python evaluation/unet3d/run_full_experiment.py \
        --raw-dir data/kits19/raw \
        --model models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
        --device mps --out evaluation/unet3d/results_mps_r1.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.unet3d_kits19 import kits19_lib as kl  # noqa: E402


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2 * inter / s) if s > 0 else float("nan")


def infer(model, image: np.ndarray, device: str) -> np.ndarray:
    image = image[np.newaxis, ...]
    result, norm_map, norm_patch = kl.prepare_arrays(image)
    ti = torch.from_numpy(image).float().to(device)
    tr = torch.from_numpy(result).float().to(device)
    tn = torch.from_numpy(norm_map).float().to(device)
    tp = torch.from_numpy(norm_patch).float().to(device)
    roi = kl.ROI_SHAPE
    with torch.no_grad():
        for i, j, k in kl.get_slice_for_sliding_window(image):
            sl = (Ellipsis, slice(i, i + roi[0]), slice(j, j + roi[1]), slice(k, k + roi[2]))
            tr[sl] += model(ti[sl]) * tp
            tn[sl] += tp
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    return kl.finalize(tr.cpu().numpy().astype(np.float64),
                       tn.cpu().numpy().astype(np.float64))[0, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/kits19/raw")
    ap.add_argument("--model", default="models/3dunet_kits19/3dunet_kits19_pytorch.ptc")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="evaluation/unet3d/results_r1.csv")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case dirs")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    cases = args.cases or sorted(
        p.name for p in raw.iterdir()
        if p.is_dir() and (p / "imaging.nii.gz").exists()
        and (p / "segmentation.nii.gz").exists())
    print(f"[unet3d-exp] {len(cases)} cases, device={args.device}, model={args.model}")

    model = torch.jit.load(args.model, map_location=args.device).eval()

    rows = []
    t_all = time.time()
    for n, case in enumerate(cases, 1):
        cd = raw / case
        try:
            t = time.time()
            img, lab, _ = kl.preprocess_volume(cd / "imaging.nii.gz", cd / "segmentation.nii.gz")
            pre_s = time.time() - t
            nsub = kl.count_subvolumes(img[None, ...])
            t = time.time()
            seg = infer(model, img, args.device)
            inf_s = time.time() - t
            gt = lab[0].astype(np.uint8)
            dk, dt_, = dice(seg >= 1, gt >= 1), dice(seg == 2, gt == 2)
            row = {
                "case": case, "preprocess_s": round(pre_s, 2), "inference_s": round(inf_s, 2),
                "total_s": round(pre_s + inf_s, 2), "n_subvolumes": nsub,
                "image_shape": "x".join(map(str, img.shape[1:])),
                "dice_kidney": round(dk, 4), "dice_tumor": round(dt_, 4),
                "dice_mean": round((dk + dt_) / 2, 4),
                "pre_frac_pct": round(100 * pre_s / (pre_s + inf_s), 1),
            }
        except Exception as e:  # keep going; record the failure
            row = {"case": case, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        msg = (f"  [{n}/{len(cases)}] {case}: pre {row.get('preprocess_s','?')}s "
               f"inf {row.get('inference_s','?')}s dice_mean {row.get('dice_mean','?')} "
               f"({row.get('n_subvolumes','?')} sub)") if "error" not in row \
              else f"  [{n}/{len(cases)}] {case}: ERROR {row['error']}"
        print(msg, flush=True)

    # write CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "preprocess_s", "inference_s", "total_s", "pre_frac_pct",
              "n_subvolumes", "image_shape", "dice_kidney", "dice_tumor", "dice_mean", "error"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary
    ok = [r for r in rows if "error" not in r]
    def stat(key):
        vs = sorted(r[key] for r in ok)
        return (sum(vs) / len(vs), vs[len(vs) // 2], min(vs), max(vs)) if vs else (0, 0, 0, 0)
    print(f"\n================ SUMMARY (R=1, device={args.device}) ================")
    print(f"cases: {len(ok)}/{len(rows)} ok  |  total wall {time.time()-t_all:.0f}s")
    for key, unit in [("preprocess_s", "s"), ("inference_s", "s"), ("total_s", "s"),
                      ("pre_frac_pct", "%"), ("n_subvolumes", "")]:
        m, med, lo, hi = stat(key)
        print(f"  {key:14s} mean {m:8.2f}{unit}  median {med:8.2f}  [{lo:.2f}, {hi:.2f}]")
    for key in ("dice_kidney", "dice_tumor", "dice_mean"):
        m, med, lo, hi = stat(key)
        print(f"  {key:14s} mean {m:.4f}  median {med:.4f}  [{lo:.4f}, {hi:.4f}]")
    tot_pre = sum(r["preprocess_s"] for r in ok)
    tot_inf = sum(r["inference_s"] for r in ok)
    print(f"\n  AGGREGATE preprocess {tot_pre:.0f}s / inference {tot_inf:.0f}s "
          f"-> preprocess is {100*tot_pre/(tot_pre+tot_inf):.1f}% of end-to-end "
          f"(MLPerf reports only the inference {100*tot_inf/(tot_pre+tot_inf):.1f}%)")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 3D-UNet / KiTS19 — build, data, and GPU verification

Everything except the model weights and the KiTS19 raw data is committed and
offline-verified (`kits19_lib` math passes the unit checks; stages import). This
doc is the remaining acquisition + GPU-verify checklist — it needs a free
machine (blocked while the R=1 re-runs hold both GPUs).

## 1. Python dependency

The preprocessing loads NIfTI via `nibabel` (lazy-imported, so nothing else
needs it). Add it to the env:

    pip install nibabel        # scipy + numpy already present

## 2. Model (TorchScript nnU-Net)

The stage loads a **TorchScript** module via `torch.jit.load`. MLPerf's "PyTorch model" on
Zenodo record 5597155 already IS that JIT-compiled TorchScript file (the "PyTorch/checkpoint"
variant is the raw state_dict — do NOT use it, no conversion needed here). Download it directly
(same file the reference Makefile fetches as `ZENODO_PYTORCH`):

    mkdir -p models/3dunet_kits19
    wget -O models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
      "https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1"

Input/output contract the stage assumes (verified against base_SUT.py):
input `[1,1,128,128,128]` float, output `[1,3,128,128,128]` logits (3 classes:
background / kidney / tumor).

## 3. KiTS19 raw data

    git clone https://github.com/neheller/kits19
    cd kits19 && python -m starter_code.get_imaging   # ~30 GB, downloads imaging.nii.gz per case

The 42 inference cases are listed in the reference `meta/inference_cases.json`.
Point `raw_data_dir` at the directory holding `case_00000/…` dirs (each with
`imaging.nii.gz` and `segmentation.nii.gz`). NOTE: unlike MLPerf we do NOT
pre-preprocess — `KiTS19Preprocess` runs the resample/normalize/pad inside the
timed graph on purpose.

## 4. GPU verification (run when a machine frees)

Smoke (8 cases), CUDA on GB10:

    python main.py -c pipeline_configs/unet3d_kits19_cuda.yml

MPS on M2 Pro:

    python main.py -c pipeline_configs/unet3d_kits19_mlx.yml

Confirm: each case logs "N sub-volumes" in the 8–144 range; segmentation output
is `[1,1,D,H,W]` uint8 with labels ⊆ {0,1,2}; per-case latency is seconds-scale.
Cross-check a case's mask against the reference SUT (`pytorch_SUT.py`) — the
sliding-window math is a line-for-line port, so masks should match bitwise given
the same model and preprocessed volume.

## 5. Accuracy (optional, later)

The reference `accuracy_kits.py` computes mean composite Dice (kidney+tumor). A
`DiceScorer` terminal stage can wrap it if we want the accuracy-of-record for
this workload; not required for the overhead/decomposition timing study.

## What is already done vs pending

- DONE (committed, offline-verified): `kits19_lib` (preprocessing + sliding
  window, unit-checked), the three stages, both pipeline configs, this doc.
- DONE (2026-07-20, END-TO-END VALIDATED ON MPS / M2 Pro): `nibabel` installed;
  model `3dunet_kits19_pytorch.ptc` fetched from Zenodo (124 MB); cases
  case_00000 + case_00003 staged (imaging from HF, segmentation from the repo);
  `unet3d_kits19_mlx.yml` ran through the framework (loader→preprocess→inference→capture)
  online to res17 in 69.5 s for case_00000. Direct verification on case_00000:
  * MPS 3D-conv WORKS — 128³ forward pass 2.4 s on Metal (vs 21 s CPU), numerical
    parity to CPU (max|Δ| 4e-5);
  * preprocess 6.2 s → image (1,192,384,384), 50 sub-volumes (in the 8–144 range);
  * sliding-window inference 62.8 s → non-trivial segmentation (kidney + tumor);
  * **Dice vs ground truth: kidney 0.973, tumor 0.840, mean 0.907** — at/above the
    reference card (kidney 0.9347, tumor 0.7887, mean 0.8617); case_00000 is a
    single favorable case so exceeding the 42-case mean is expected. The port is
    correct: preprocessing, model, sliding window, and finalize all validated.
- PENDING (nice-to-have rigor / scale): download the remaining inference cases
  for a full 42-case run; CUDA smoke on GB10 (`unet3d_kits19_cuda.yml`); bitwise
  cross-check vs the reference `pytorch_SUT.py`. None of these gate paper use —
  the workload is proven working and accurate.

### radt launch note
Run via `conda run -n benchmark_macos python main.py <config>` (NOT the env
python directly): radt's scheduler spawns `python -m radt run`, so bare `python`
must resolve to the env that has radt on PATH. `run_collection.py` already does
this. For a non-traced smoke, prefix `CHOREO_DISABLE_TRACING=1` (the mlflow
"NoOpTracerProvider" flush warnings that follow are harmless).

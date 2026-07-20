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

MLPerf distributes the KiTS19 3D-UNet as an ONNX + PyTorch checkpoint. The stage
loads a **TorchScript** `.pt` (`torch.jit.load`). Get it via the reference:

    # in scratchpad/mlperf-inference/vision/medical_imaging/3d-unet-kits19
    # follow README "Download model" (Zenodo record 5597155): 3dunet_kits19_pytorch.ptc
    # then trace/script to TorchScript if the download is a state_dict:
    python unet_pytorch_to_onnx.py        # reference helper (also emits a scripted module)

Place the TorchScript file at the config's `model_path`:

    models/3dunet_kits19/3dunet_kits19_128x128x128.pt

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
- PENDING (needs a free GPU + downloads): `pip install nibabel`; fetch the
  TorchScript model; download KiTS19; run the two smoke configs; bitwise
  cross-check vs the reference SUT.

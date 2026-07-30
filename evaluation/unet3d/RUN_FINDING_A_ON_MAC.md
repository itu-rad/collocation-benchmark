# Finding A — 3D-UNet preprocessing breakdown, on a Mac

Runs MLPerf's own 3D-UNet/KiTS19 inference workload **end-to-end** (preprocess → sliding-window
inference) over the 42 studies and records **per-stage latency**: `preprocess_s` (CPU — the part
MLPerf excludes from its timed region) and `inference_s` (GPU — the only part MLPerf times). The
result shows the excluded preprocessing is (1) **sample-dependent** and (2) **a larger share of
end-to-end on a faster GPU** (Amdahl). Run on an **idle** machine — this is a single-request
latency measurement, so background load contaminates it.

## 1. One-time setup

```bash
conda env create -f environments/macos.yaml     # creates env `benchmark_macos` (now includes nibabel + scipy)
conda activate benchmark_macos
# fallback if the env predates the nibabel/scipy add:  pip install nibabel scipy
```

## 2. Get the model (~124 MB, once) — direct download, ready to use

MLPerf's "PyTorch model" on Zenodo record 5597155 **is** the JIT-compiled TorchScript module
(`torch.jit.load`-able) — nothing to convert. Pull it straight to the path the run command expects
(this is the exact file the reference harness Makefile fetches as `ZENODO_PYTORCH`):

```bash
mkdir -p models/3dunet_kits19
wget -O models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
  "https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1"
```

Contract: input `[1,1,128,128,128]` float → output `[1,3,128,128,128]` logits (background/kidney/tumor).

## 3. Get the KiTS19 raw data (~30 GB download, once)

```bash
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt            # deps for the imaging downloader
python3 -m starter_code.get_imaging         # downloads imaging.nii.gz into data/case_XXXXX/
```

This yields ~210 `<kits19>/data/case_XXXXX/` dirs, each with `imaging.nii.gz` (just downloaded)
and `segmentation.nii.gz` (already in the clone). **Point `--raw-dir` at `<kits19>/data`** and
select the inference subset with `--cases-file` (step 4) — the experiment runs exactly the **42
cases** in `evaluation/unet3d/inference_cases.json`, not all ~210. We do **not** pre-preprocess —
the resample/normalize/pad runs inside the timed path on purpose.

(MLPerf's own harness pads this set to 43 by duplicating case_00185 as case_00400; that copy is a
redundant identical case, so we run the 42 unique ones.)

## 4. Run the experiment (~1 h for 42 cases on an M2)

```bash
# Quick smoke first (3 small studies, ~1 min) to confirm the setup works:
python evaluation/unet3d/run_full_experiment.py \
    --raw-dir <kits19>/data --model models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
    --device mps --cases case_00160 case_00138 case_00112 --out /tmp/smoke.csv

# Full run — the 42 inference cases, written to the exact path the analyzer reads:
python evaluation/unet3d/run_full_experiment.py \
    --raw-dir <kits19>/data --model models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
    --cases-file evaluation/unet3d/inference_cases.json \
    --device mps --out evaluation/unet3d/results_mps_r1.csv
```

Per case it prints `pre … s / inf … s / dice_mean …` and writes the CSV. Dice should land near the
reference model card (mean ~0.86) — that is the built-in accuracy validation (same model, same
data as MLPerf), so no separate MLPerf harness run is needed.

## 5. Analyze — figures + the CPU/GPU stage breakdown

```bash
python evaluation/unet3d/analyze_preprocessing.py --fig evaluation/unet3d/preprocessing_fraction.pdf
```

Prints, and writes two figures:
- **Per-device summary** + **cross-device amplification** (GPU speedup vs CPU-preprocess speedup →
  preprocessing share) + the **per-stage CPU vs GPU latency breakdown table**.
- `preprocessing_fraction.pdf/png` — preprocessing share vs study size (the sample-dependence +
  the cross-device shift).
- `preprocessing_fraction_stages.pdf/png` — per-study **stacked bars of CPU (preprocess) + GPU
  (inference)** time, both devices, studies sorted by size.

The analyzer reads **both** `results_mps_r1.csv` (your clean-Mac run) and `results_cuda_r1.csv`
(the GB10 run, shipped in the repo) so the figures are the **cross-device** comparison — your Mac
run supplies the mps points; the committed cuda CSV supplies the GB10 points. To refresh the GB10
half, re-run step 4 with `--device cuda --out evaluation/unet3d/results_cuda_r1.csv` on GB10.

## What it shows (Finding A)
- **Sample-dependent:** preprocessing share falls with study size (inference scales with subvolume
  count 8–144; preprocessing is ~flat), so small studies are preprocessing-dominated (up to ~70%
  on GB10).
- **Amplifies on the faster GPU:** the GPU is ~10× faster while CPU preprocessing is only ~2×
  faster, so the excluded stage's share quadruples (Mac ~5% → GB10 ~19%) — and grows as
  accelerators improve.

# Conda environments (for reference)

Two collection environments, one per device. Both are built from the checked-in YAMLs and both must
use the **new bulk+proc radt** for collection (see `EXPERIMENT_PLAN.md` P0). Captured 2026-08-17.

| Env | Device | Host | Built from |
|---|---|---|---|
| `benchmark_macos` | M2 Pro (mlx / mps) | local Mac | `environments/macos.yaml` |
| `benchmark_nvidia` | GB10 / DGX Spark (cuda) | `babyxena` (ssh, `-i ~/.ssh/id_rsa`) | `environments/nvidia.yaml` |

## Package versions (2026-08-17)

**benchmark_macos** — python 3.10.20, macOS-26.5.2-arm64:
`torch==2.10.0`, `transformers==5.2.0`, `mlx==0.32.0`, `numpy==2.2.6`, `radt==0.2.29`,
`mlflow==3.10.0`.
⚠️ Its `radt` is currently **broken** (editable install pointed at a deleted session scratchpad —
`radt.run` won't import). **P0 must reinstall the bulk radt here** (which also bumps mlflow → 3.15.1).

**benchmark_nvidia (GB10)** — python 3.10.19, Linux-aarch64 (glibc 2.39):
`torch==2.10.0+cu130`, `transformers==5.2.0`, `numpy==2.2.6`, `radt==0.2.29`, `mlflow==3.15.1`.
Already running the **bulk radt** (editable install from `/home/roba/radt-bulk`).

## radt (the tracing layer)

- **Use the bulk+proc radt for all collection:** `github.com/itu-rad/radt` branch
  `feat/proc-owned-bulk-tracing` (currently `0b497f6` "ensure tracing in multiprocess, verify spans").
  Spools all spans into one gzipped artifact per run (`radt-trace/spans-*.jsonl.gz` + `manifest.json`)
  → negligible workload overhead, 100% capture.
- **P0 pins it at a fixed tag and editable-installs it into BOTH envs** (`pip install -e <checkout>/radt`),
  replacing the old `@9dda7b8`. It pulls `mlflow>=3.15`.
- Do **not** use the older radt (`@9dda7b8` async-tracing, or the proc-only PR fork at
  `~/Documents/work/research/radt`) — superseded.
- the framework switches it on with `CHOREO_PROC_TRACE=1` (set in each `collect.sh`).
- Carried local patch: `evaluation/radt-patches/0001-amc-bandwidth-listener.patch` (Apple AMC
  per-engine bandwidth listener — needed for E5 attribution on the M2). Apply it to the radt checkout
  before installing on macOS.

## MLflow / res17 (the tracking server)

- Server: `https://res17.itu.dk` (the RAD group's MLflow). Real collection experiment id: **138**.
  Throwaway prototype experiment: **142**.
- **Credentials are NEVER in the repo.** They live as **conda env config vars** on each env:
  `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
  (`conda env config vars set MLFLOW_TRACKING_URI=... -n benchmark_macos`, etc.). These are lost when
  an env is rebuilt — re-set them after any `conda env create`.

## Rebuild from scratch

```
conda env create -f environments/macos.yaml     # -> benchmark_macos   (M2)
conda env create -f environments/nvidia.yaml     # -> benchmark_nvidia  (GB10)
# then, in each env:
pip install -e <bulk-radt-checkout>/radt         # the pinned itu-rad/radt bulk branch
conda env config vars set MLFLOW_TRACKING_URI=https://res17.itu.dk \
    MLFLOW_TRACKING_USERNAME=<user> MLFLOW_TRACKING_PASSWORD=<pass> -n <env>
```
The collection drivers hard-abort on a radt/torch mismatch (`check_environment`), so always rebuild
from the YAML — never trust a pre-existing env.

## Notes

- Retired: `benchmark_macos_overhead` (old workaround env; do not use).
- The GB10 bulk-radt checkout lives at `/home/roba/radt-bulk`; the the repo on GB10 is
  `/home/roba/collocation-benchmark`.

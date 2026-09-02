# Hyperparameter pilots + knob provenance

Every experiment-config knob (arrival rate, max_queries, queue_depth, warm-up
drop k, R, timeout) derives from a **pre-registered rule** whose inputs come
from short **serial pilot runs** measuring service time and warm-up horizon per
(workload, device). This package runs the pilots, derives the knobs, locks them
into the committed configs, renders the paper's knob tables, and verifies the
rules held after collection. Protocol: `PAPER_TODO.md` §2.6; rule registry:
`derive_knobs.py` docstring.

Pilots are excluded from reported data and **commit-pinned**: `pilot_env.txt`
records the git SHA + machine SKU; do not mix pilot CSVs across commits.

## Pipeline

```
run_pilots.py --device {mlx,cuda}      # serial pilots -> results/pilot_*.csv
derive_knobs.py [--variant ...]        # rules + pilots -> knobs.yml
apply_knobs.py --dry-run               # review diffs, then run without flag
generate_knob_tables.py > knob_tables.tex
...collect the real experiments...
verify_knobs.py --traces '...' --experiment e4 --device m2pro
```

`knobs.yml` is the single provenance artifact (committed). Drivers/analyzers
read non-config knobs (R, warm-up k, quality N) via
`pilot_lib.load_knobs()/get_knob()`.

Variants (pending the E3′/E6′ redesign sign-off — both rule bindings are
pre-registered): `--variant e3=vqa|dose_response --variant
e6=torchvision|rag_indexing`. Flipping a variant re-derives; no machine time
is re-spent (pilots measure workload *components*, shared by both variants).

## GB10 spec (Ties) — run verbatim

Prereqs: `benchmark_nvidia` env **built from `environments/nvidia.yaml`** —
NOT any pre-existing env (on babyxena the old `benchmark` env had radt 0.2.23
and a CPU-only torch; found 2026-07-13). The yaml pins torch 2.10.0+cu130,
transformers 5.2.0, and radt 0.2.29 (async_tracing @3ba61cb). The drivers now
**hard-abort** on a mismatched env (`check_environment`). Verify first:

```bash
conda run -n benchmark_nvidia python -c \
  "import radt, torch; print(radt.__version__, torch.__version__, torch.cuda.is_available())"
# expect: 0.2.29 2.10.0+cu130 True
```

Models + datasets pre-fetched (first pilot run downloads otherwise — fine, but
then re-run that cell with `--force` so no download is timed), idle box.

```bash
git fetch && git checkout feat/paper-hardening && git pull
conda activate benchmark_nvidia
python evaluation/pilots/run_pilots.py --list            # sanity: see cells
python evaluation/pilots/run_pilots.py --device cuda     # ~1-1.5 h
# then EITHER commit evaluation/pilots/results/ + pilot_env.txt and push,
# OR run the derivation here:
python evaluation/pilots/derive_knobs.py
python evaluation/pilots/apply_knobs.py --dry-run        # review
python evaluation/pilots/apply_knobs.py
git add -A evaluation/pilots pipeline_configs evaluation/self_rag/configs
git commit -m "GB10 pilot knobs"
```

Wall-time expectations (BF16 9B is the slow item): e4 factoid cells ~5–15 min
total, multihop ~10–20 min, the rest minutes each. **Abort criteria:** any
cell exceeding its per-cell timeout twice; `pilot_env.txt` showing a `-dirty`
commit; or thermal throttling (log `nvidia-smi -q -d TEMPERATURE` before/after).

**The main GB10 collection pass starts only after the cuda knobs are
committed** — configs carry `[[pending pilot]]`-derived placeholder values
until then.

## The idle-M2 session (author)

Canonical M2 collection env: **`benchmark_macos`**, rebuilt from
`environments/macos.yaml` (2026-07-13: the yaml pins radt 0.2.29 async_tracing;
the old `benchmark_macos_overhead` workaround env is retired).

1. Machine idle (close apps, disable heavy background jobs), on AC power.
2. `python evaluation/pilots/run_pilots.py --device mlx` (~45–90 min).
3. Same session, E2 re-collection:
   `bash evaluation/overheads/modularity_overhead/collect_e2.sh m2pro 11`.
4. `derive_knobs.py` → `apply_knobs.py --dry-run` → review → apply → commit.

## Notes

- Pilot cells override each base config's loadgen block to the closed-loop
  `OfflineLoadScheduler` (one query in flight = serial service time) and run
  with `CHOREO_DISABLE_TRACING=1` (core dispatch only).
- Cells marked `BLOCKED` by `--list` need apparatus from
  `evaluation/contention/contention.md` (C3 memory-streaming stage,
  EmbedStage/ChromaIndexer) and are excluded from the session.
- Open-loop collection runs now emit an `<label>_arrivals.csv` sidecar
  (intended vs actual submit time + put-block seconds) — `verify_knobs.py`
  uses it to prove queue_depth never blocked and the realized rate matched λ.
- Unit tests: `python evaluation/pilots/test_pilots.py`.

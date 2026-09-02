# Staging directory — empty at rest

`main.py` hardcodes its per-run CSV/JSONL output here (`log_dir = "evaluation/results"`).
This directory is **transient**: each collection harness moves its runs out of here into the
results folder of the experiment that produced them, so results always live beside their
experiment.

| experiment | results live in |
|---|---|
| E1 framework overhead | `overheads/framework_overhead/results/<machine>/` |
| E2 modularity overhead | `overheads/modularity_overhead/results/` |
| E3 MLPerf 3D-UNet | `unet3d/results/` |
| §5.1 Self-RAG | `self_rag/results/` |

If files accumulate here, a run was made outside a harness (a smoke test or a probe) — they
are not results of record and can be deleted.

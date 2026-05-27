# Self-RAG: monolith vs decomposition

A case study comparing two topologies for an agentic Self-RAG pipeline on
identical work:

| Topology | Models | Per query |
|---|---|---|
| **Monolith** | 1× 9B (`Qwen/Qwen3.5-9B`, MLX: `Qwen3.5-9B-OptiQ-4bit`) | one structured call that grades + answers + self-checks |
| **Decomposed** | 3× 4B (`Qwen/Qwen3.5-4B`) — grader · generator · hallucination-checker (rewriter shares the grader) | grade → generate → hallucination-check → optional rewrite + retry |

The question: is one large model that does everything better than several
smaller specialized models, one per sub-task? Findings (DGX Spark / GB10) are in
[`monolith_vs_decomposed_dgxspark.md`](monolith_vs_decomposed_dgxspark.md); the
decomposed pipeline's wiring is in [`topology_diagram.md`](topology_diagram.md).

## Layout

```
evaluation/self_rag/
├── README.md                            # this file
├── monolith_vs_decomposed_dgxspark.md   # results + analysis
├── topology_diagram.md / .png / .svg    # decomposed-pipeline flowchart
└── configs/
    ├── factoid_monolith_mlx.yml         # easy task (rag-mini-wikipedia), Apple Silicon
    ├── factoid_decomposed_mlx.yml
    ├── factoid_monolith_cuda.yml        # easy task, DGX Spark / CUDA
    ├── factoid_decomposed_cuda.yml
    ├── multihop_monolith_cuda.yml       # hard task (HotpotQA), DGX Spark / CUDA
    └── multihop_decomposed_cuda.yml
```

The pipeline stages live in `stages/self_rag/` (dataloader, retriever, monolith
formatter/router, plus the per-sub-task formatters and routers).

## Experiment index

| Experiment | Variant | Backend | Config |
|---|---|---|---|
| Factoid (`rag-mini-wikipedia`) | Monolith | MLX | `configs/factoid_monolith_mlx.yml` |
| Factoid | Decomposed | MLX | `configs/factoid_decomposed_mlx.yml` |
| Factoid | Monolith | CUDA | `configs/factoid_monolith_cuda.yml` |
| Factoid | Decomposed | CUDA | `configs/factoid_decomposed_cuda.yml` |
| Multi-hop (`HotpotQA`) | Monolith | CUDA | `configs/multihop_monolith_cuda.yml` |
| Multi-hop | Decomposed | CUDA | `configs/multihop_decomposed_cuda.yml` |

## Setup

Run everything from the repo root with the matching conda environment active:

- **CUDA / DGX Spark:** `conda activate benchmark_nvidia`
- **MLX / Apple Silicon:** `conda activate benchmark`

Models download from HuggingFace on first run. The runner CLI:

```
python main.py <config> -p 0 [--serialize true|false] [--label <name>]
```

`-p 0` selects the (only) pipeline. `--serialize false` is pipelined (queries
overlap), `true` is one-at-a-time. `--label` sets the output-file stem under
`evaluation/results/` (timing CSV `<label>.csv` and `<label>_outputs.jsonl`);
without it the stem defaults to the pipeline name lowercased.

## Reproduce

### Smoke test (quality, 5 queries)

The semantic verifier reads `self_rag_monolith_outputs.jsonl` /
`self_rag_decomposed_outputs.jsonl`, so label the smoke runs to match:

```bash
python main.py evaluation/self_rag/configs/factoid_monolith_cuda.yml   -p 0 --label self_rag_monolith
python main.py evaluation/self_rag/configs/factoid_decomposed_cuda.yml -p 0 --label self_rag_decomposed
python evaluation/scripts/verify_complex_cases.py        # -> evaluation/results/verification_report.md
```

Expect ~4/5 golden-answer hits per pipeline. (For a quick 5-query pass, drop
`max_queries` in the config; the committed configs run 30.)

### Factoid 2×2 (CUDA)

```bash
python main.py evaluation/self_rag/configs/factoid_monolith_cuda.yml   -p 0 --serialize false --label self_rag_factoid_monolith_pipe
python main.py evaluation/self_rag/configs/factoid_monolith_cuda.yml   -p 0 --serialize true  --label self_rag_factoid_monolith_serial
python main.py evaluation/self_rag/configs/factoid_decomposed_cuda.yml -p 0 --serialize false --label self_rag_factoid_decomposed_pipe
python main.py evaluation/self_rag/configs/factoid_decomposed_cuda.yml -p 0 --serialize true  --label self_rag_factoid_decomposed_serial

python evaluation/scripts/bandwidth_analysis.py \
  self_rag_factoid_monolith_pipe self_rag_factoid_monolith_serial \
  self_rag_factoid_decomposed_pipe self_rag_factoid_decomposed_serial \
  --out evaluation/results/self_rag_factoid_bandwidth.md
```

### Multi-hop 2×2 (CUDA, HotpotQA)

```bash
python main.py evaluation/self_rag/configs/multihop_monolith_cuda.yml   -p 0 --serialize false --label self_rag_multihop_monolith_pipe
python main.py evaluation/self_rag/configs/multihop_monolith_cuda.yml   -p 0 --serialize true  --label self_rag_multihop_monolith_serial
python main.py evaluation/self_rag/configs/multihop_decomposed_cuda.yml -p 0 --serialize false --label self_rag_multihop_decomposed_pipe
python main.py evaluation/self_rag/configs/multihop_decomposed_cuda.yml -p 0 --serialize true  --label self_rag_multihop_decomposed_serial

python evaluation/scripts/bandwidth_analysis.py \
  self_rag_multihop_monolith_pipe self_rag_multihop_monolith_serial \
  self_rag_multihop_decomposed_pipe self_rag_multihop_decomposed_serial \
  --out evaluation/results/self_rag_multihop_bandwidth.md
```

The four positional labels map to the 2×2 cells in order
(monolith-pipe, monolith-serial, decomposed-pipe, decomposed-serial).

### Factoid on Apple Silicon (MLX)

Same as the factoid CUDA block, swapping the configs for the `*_mlx.yml`
variants (and `conda activate benchmark`). The MLX configs run the original
smoke settings (5 queries); bump `max_queries`/`rate` to match the 30-query
CUDA runs if comparing head-to-head.

## Outputs

All runs write to `evaluation/results/` (gitignored): `<label>.csv` (per-stage
timing trace) and `<label>_outputs.jsonl` (per-query question / retrieved docs /
answer). The analysis scripts read those stems and write Markdown reports there.

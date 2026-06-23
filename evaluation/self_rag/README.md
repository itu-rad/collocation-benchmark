# Self-RAG: monolith vs decomposition

A case study comparing topologies for an agentic Self-RAG pipeline on identical
work. There are four arms:

| Topology | Models / instances | Per query |
|---|---|---|
| **Monolith 9B** | 1× 9B (`Qwen/Qwen3.5-9B`, MLX: `Qwen3.5-9B-OptiQ-4bit`) | one structured call that grades + answers + self-checks |
| **Monolith 4B** | 1× 4B (`Qwen/Qwen3.5-4B`, MLX: `Qwen3.5-4B-OptiQ-4bit`) | same single call, size-matched control (isolates model size from topology) |
| **Decomposed** | 3× 4B *separate instances* — grader · generator · hallucination-checker (rewriter shares the grader) | grade → generate → hallucination-check → optional rewrite + retry |
| **Decomposed-Shared** | 1× 4B, *all stages share one resident instance* (`depends_on_id`, one mutex) | same decomposed graph + narrow per-role prompts, but logical decomposition only |

The questions these arms isolate:

- **Monolith vs decomposition** (9B single call vs 3× 4B specialists): is one
  large do-everything model better than several smaller specialized ones?
- **Physical vs logical decomposition** (Decomposed vs Decomposed-Shared): what
  do the extra instances buy (cross-query overlap) at the cost of ~3× model
  memory, vs. just splitting one call into three narrow calls on one model?
- **Size vs topology** (Monolith 9B vs Monolith 4B): how much of the monolith's
  behavior is model size rather than its single-call structure?
- **Engine comparison** (HF transformers vs vLLM vs Ollama): the single-instance
  arms (Monolith 4B, Decomposed-Shared) are served three ways on *identical 4B
  weights* to compare inference backends.

Findings (DGX Spark / GB10) are in
[`monolith_vs_decomposed_dgxspark.md`](monolith_vs_decomposed_dgxspark.md) (M2
Pro in [`monolith_vs_decomposed_m2pro.md`](monolith_vs_decomposed_m2pro.md)); the
decomposed pipeline's wiring is in [`topology_diagram.md`](topology_diagram.md).

## Backends

Each topology has configs for one or more inference backends:

| Suffix | Backend | Platform |
|---|---|---|
| `_cuda` | HF Transformers (in-process) | DGX Spark / GB10, CUDA |
| `_mlx` | MLX (in-process) | Apple Silicon |
| `_vllm_cuda` | local vLLM server (OpenAI-compatible HTTP, continuous batching) via litellm | CUDA |
| `_ollama` | local Ollama server (HTTP) via litellm | cross-platform (NVIDIA + Apple Silicon) |

## Layout

```
evaluation/self_rag/
├── README.md                              # this file
├── monolith_vs_decomposed_dgxspark.md     # results + analysis (DGX Spark / GB10)
├── monolith_vs_decomposed_m2pro.md        # results + analysis (M2 Pro)
├── topology_diagram.md / .png / .svg      # decomposed-pipeline flowchart
└── configs/
    ├── factoid_monolith_cuda.yml          # easy task (rag-mini-wikipedia), 9B
    ├── factoid_monolith_mlx.yml
    ├── factoid_monolith_4b_cuda.yml       # size-matched 4B monolith control
    ├── factoid_monolith_4b_mlx.yml
    ├── factoid_monolith_4b_vllm_cuda.yml  # 4B monolith, vLLM server
    ├── factoid_monolith_4b_ollama.yml     # 4B monolith, Ollama server
    ├── factoid_decomposed_cuda.yml        # 3× 4B separate instances
    ├── factoid_decomposed_mlx.yml
    ├── factoid_decomposed_shared_cuda.yml # 1× 4B shared (logical decomposition)
    ├── factoid_decomposed_shared_mlx.yml
    ├── factoid_decomposed_shared_vllm_cuda.yml  # shared 4B, vLLM server
    ├── factoid_decomposed_shared_ollama.yml     # shared 4B, Ollama server
    ├── multihop_monolith_cuda.yml         # hard task (HotpotQA), 9B
    ├── multihop_monolith_mlx.yml
    ├── multihop_decomposed_cuda.yml
    └── multihop_decomposed_mlx.yml
```

The pipeline stages live in `stages/self_rag/` (dataloader, retriever, monolith
formatter/router, plus the per-sub-task formatters and routers).

## Experiment index

| Experiment | Variant | Backend | Config |
|---|---|---|---|
| Factoid (`rag-mini-wikipedia`) | Monolith 9B | CUDA | `configs/factoid_monolith_cuda.yml` |
| Factoid | Monolith 9B | MLX | `configs/factoid_monolith_mlx.yml` |
| Factoid | Monolith 4B | CUDA | `configs/factoid_monolith_4b_cuda.yml` |
| Factoid | Monolith 4B | MLX | `configs/factoid_monolith_4b_mlx.yml` |
| Factoid | Monolith 4B | vLLM | `configs/factoid_monolith_4b_vllm_cuda.yml` |
| Factoid | Monolith 4B | Ollama | `configs/factoid_monolith_4b_ollama.yml` |
| Factoid | Decomposed (3× 4B) | CUDA | `configs/factoid_decomposed_cuda.yml` |
| Factoid | Decomposed (3× 4B) | MLX | `configs/factoid_decomposed_mlx.yml` |
| Factoid | Decomposed-Shared (1× 4B) | CUDA | `configs/factoid_decomposed_shared_cuda.yml` |
| Factoid | Decomposed-Shared (1× 4B) | MLX | `configs/factoid_decomposed_shared_mlx.yml` |
| Factoid | Decomposed-Shared (1× 4B) | vLLM | `configs/factoid_decomposed_shared_vllm_cuda.yml` |
| Factoid | Decomposed-Shared (1× 4B) | Ollama | `configs/factoid_decomposed_shared_ollama.yml` |
| Multi-hop (`HotpotQA`) | Monolith 9B | CUDA | `configs/multihop_monolith_cuda.yml` |
| Multi-hop | Monolith 9B | MLX | `configs/multihop_monolith_mlx.yml` |
| Multi-hop | Decomposed (3× 4B) | CUDA | `configs/multihop_decomposed_cuda.yml` |
| Multi-hop | Decomposed (3× 4B) | MLX | `configs/multihop_decomposed_mlx.yml` |

## Setup

Run everything from the repo root with the matching conda environment active:

- **CUDA / DGX Spark** (HF Transformers): `conda activate benchmark_nvidia`
- **MLX / Apple Silicon:** `conda activate benchmark`
- **vLLM / Ollama** (`_vllm_cuda` / `_ollama` configs): `conda activate benchmark_engines`
  (Ollama binary is at `~/.local/bin`; make sure it's on `PATH`)

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

### Topology controls (CUDA)

The 4B monolith and the shared-instance decomposition isolate model size and
physical vs. logical decomposition. Run them the same way as the 2×2 arms:

```bash
python main.py evaluation/self_rag/configs/factoid_monolith_4b_cuda.yml       -p 0 --serialize false --label self_rag_factoid_monolith4b_pipe
python main.py evaluation/self_rag/configs/factoid_decomposed_shared_cuda.yml -p 0 --serialize false --label self_rag_factoid_shared_pipe
```

Compare against `self_rag_factoid_monolith_pipe` (size: 9B vs 4B) and
`self_rag_factoid_decomposed_pipe` (3 instances vs 1 shared) via
`bandwidth_analysis.py`.

### Engine comparison (identical 4B weights)

The single-instance arms are served by three backends — HF Transformers
(`_cuda`), vLLM (`_vllm_cuda`), and Ollama (`_ollama`) — on the same Qwen3.5-4B
weights, to compare inference engines under identical work. Run under
`benchmark_engines`:

```bash
for engine in cuda vllm_cuda ollama; do
  python main.py evaluation/self_rag/configs/factoid_monolith_4b_${engine}.yml \
    -p 0 --serialize false --label self_rag_factoid_monolith4b_${engine}
  python main.py evaluation/self_rag/configs/factoid_decomposed_shared_${engine}.yml \
    -p 0 --serialize false --label self_rag_factoid_shared_${engine}
done
```

(The `_cuda` configs are HF Transformers; `_vllm_cuda`/`_ollama` launch their
server in-process.)

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

### Apple Silicon (MLX)

Every factoid and multi-hop arm has an `*_mlx.yml` counterpart (including the
`monolith_4b` and `decomposed_shared` controls). Run the same blocks above with
the `_mlx` configs and `conda activate benchmark`. The MLX configs run smaller
smoke settings (10 queries); bump `max_queries`/`rate` to match the 30-query
CUDA runs if comparing head-to-head. The cross-platform Ollama variants also run
here (`conda activate benchmark_engines`).

## Outputs

All runs write to `evaluation/results/` (gitignored): `<label>.csv` (per-stage
timing trace) and `<label>_outputs.jsonl` (per-query question / retrieved docs /
answer). The analysis scripts read those stems and write Markdown reports there.

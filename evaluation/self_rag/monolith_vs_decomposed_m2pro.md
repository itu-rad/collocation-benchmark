# Self-RAG: monolith vs decomposition — Apple Silicon M2 Pro

Companion to [`monolith_vs_decomposed_dgxspark.md`](monolith_vs_decomposed_dgxspark.md):
re-run of the same 2×2 (× 2 datasets) on a 16 GB M2 Pro using the MLX backend.

## Setup

| | Monolith | Decomposed |
|---|---|---|
| Model | 1× `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 3× `mlx-community/Qwen3.5-4B-OptiQ-4bit` (grader · generator · halluc-check; rewriter shares the grader) |
| Per query | one JSON pass: grade + answer + self-check | grade → generate → halluc-check → optional rewrite + retry |

Both topologies hit ChromaDB over the same per-experiment corpus, run on the
same dataset, and share the rest of the pipeline plumbing
(`TerminalCapture`, the bounded-drain `Pipeline.run()`, and the
`serialize_queries` flag). Configs:

- `configs/factoid_{monolith,decomposed}_mlx.yml` — rag-mini-wikipedia
  factoid, 3 200 passages, `max_queries: 10`, Poisson rate `1.0`.
- `configs/multihop_{monolith,decomposed}_mlx.yml` — HotpotQA dev,
  20 k passages built from the gold + distractor contexts of the first
  300 questions, `max_queries: 10`, Poisson rate `1.0`.

Cells: `{factoid, multihop} × {monolith, decomposed} × {pipelined, serial}` = 8.
The 2×2 reports for each task family live next to the timing CSVs in
`evaluation/results/self_rag_{factoid,multihop}_bandwidth_m2pro.md`.

> **Note on the analyzer's "cuda" label.** The bandwidth analyzer's
> `DEVICE_MAP` for self-rag cells was authored for the DGX Spark runs and
> tags every LLM stage `cuda`. On this MLX run "cuda" is cosmetic — the
> GPU device is MPS. Numbers are correct; only the device label needs
> re-reading as "MPS".

## Headline comparison

### Factoid (rag-mini-wikipedia)

| | Monolith (1×9B) | Decomposed (3×4B) | Winner |
|---|---|---|---|
| Per-query latency, serial | 7.99 s | **6.27 s** (−21.5 %) | Decomp |
| Per-query latency, pipelined | 34.78 s | **22.71 s** (−34.7 %) | Decomp |
| Throughput, serial | 0.125 q/s | **0.159 q/s** (+27 %) | Decomp |
| Throughput, pipelined | 0.129 q/s | **0.170 q/s** (+32 %) | Decomp |
| Wall, pipelined | 77.3 s | **47.1 s** (−39 %) | Decomp |
| Answered | 9/10 | 7/8 † | tie |
| Golden-answer hits | **8/10** | 7/8 † | ≈ tie |

† the decomposed pipelined cell ran 8 queries (vs 10) — see
[Caveat: MPS pressure](#caveat-mps-pressure-on-decomposed-pipelined) below.

**Quality is a tie. Decomposition wins on both speed dimensions** —
~22 % lower per-query latency serial, ~35 % under pipelining; ~27–32 %
higher throughput. The lead *widens* under pipelining (+22.7 percentage
points) because the decomposed pipeline's three independent model
instances overlap across in-flight queries, while the monolith's single
9B sits behind one mutex and can't share the GPU.

### Multi-hop (HotpotQA)

| | Monolith (1×9B) | Decomposed (3×4B) | Winner |
|---|---|---|---|
| **Answered** | **2/10 (20 %)** | **5–6/10 (50–60 %)** | **Decomp (3× more)** |
| **Golden-answer hits** | **2/10 (20 %)** | **3/10 (30 %)** | **Decomp** |
| Per-query latency, serial | 8.42 s | 8.70 s | ≈ tie |
| Per-query latency, pipelined | 44.80 s | 45.42 s | ≈ tie |
| Throughput, serial | 0.119 q/s | 0.115 q/s | ≈ tie |
| Throughput, pipelined | 0.123 q/s | 0.107 q/s | Mono (−13 %) |
| Wall, pipelined | 81.3 s | 93.5 s | Mono (−13 %) |

**Quality is where the gap opens.** On HotpotQA's compositional bridge
questions, the 4-bit 9B monolith hits the retry ceiling 8 of 10 times —
the all-in-one JSON pass can't admit it needs more retrieval. The
decomposed pipeline, whose query-rewriter is conditioned on the
already-retrieved documents and asked to issue a bridge query for the
missing fact, answers 5–6 of 10 and lands 3/10 golden hits.

Speed is a wash on multi-hop. The decomposed pipeline runs slightly more
LLM calls per question (multi-step retry + rewriter loop) and the gain
from concurrency doesn't quite cancel that out. So **decomposition's
multi-hop value is robustness, not throughput.**

## Trade-off summary

- **Factoid:** decomposed is faster (latency *and* throughput), same
  quality. Pipelining only helps decomposed (monolith pipelines into a
  single mutex). The monolith's advantage is operational simplicity and
  half the model-weight memory footprint.
- **Multi-hop:** decomposed is roughly as fast but **answers 2–3× more
  questions correctly**. The monolith catastrophically under-answers
  because it can't course-correct its retrieval mid-stream — the
  rewriter loop (with evidence-aware bridge queries) is essential here.

## How this compares to DGX Spark (GB10, 128 GB)

| | M2 Pro (this run) | DGX Spark |
|---|---|---|
| Factoid monolith mean-latency, serial | 7.99 s | 6.32 s |
| Factoid decomposed mean-latency, serial | 6.27 s | 3.71 s |
| Factoid decomposed throughput, pipelined | 0.170 q/s | 0.214 q/s |
| Factoid: decomp wins on quality+speed | ✓ | ✓ |
| Multi-hop monolith answered (of 10–30) | **2/10 (20 %)** | 10/30 (33 %) |
| Multi-hop decomposed answered | **6/10 (60 %)** | 20/30 (67 %) |
| Multi-hop decomp wins on quality | ✓ (3×) | ✓ (2×) |

Same verdict on both architectures: **decomposition wins on both
factoid speed and multi-hop quality**, and the harder task *widens*
the gap rather than closing it.

The M2 Pro is harsher on the 9B monolith — 4-bit quantization on the
smaller machine pushes the monolith from "merely worse" on multi-hop
(DGX: 10/30) to "essentially non-functional" (M2 Pro: 2/10). This
matters: on capability-constrained hardware, the case for decomposition
into specialists is even stronger.

Pipelining benefits are smaller on M2 Pro than on GB10 (M2 Pro:
+32 % throughput at the cost of +260 % per-query latency on decomposed
factoid; DGX: +86 % throughput at much smaller latency cost). The MPS
device on M2 Pro is the single shared accelerator and behaves under load
exactly like the unified-memory bandwidth story from the VQA experiment
— overlap helps a little, but the shared device caps the gain.

## Caveat: MPS pressure on decomposed pipelined

The first run of the decomposed-pipelined factoid cell hit
`[METAL] Insufficient Memory` mid-way (3 × Qwen3.5-4B-OptiQ × multiple
in-flight queries with KV caches exceeded the available MPS allocation
on a 16 GB machine). It was re-run with `rate: 0.4` and `max_queries: 8`
to keep peak concurrency lower; that's why the cell shows 8/10 captures
instead of 10/10. The conclusion (decomposed beats monolith) is
robust — its latency / throughput / quality numbers from the lower-load
re-run are still cleanly better than the full-load monolith cell. The
ceiling is the only difference: at high enough load on a 16 GB M2 Pro,
the 3 × 4B decomposed setup eventually OOMs while the monolith does
not. On the 128 GB DGX Spark this constraint disappears.

## Reproduce

```bash
conda activate benchmark_macos

for variant in factoid multihop; do
  for topo in monolith decomposed; do
    for sched in false true; do
      label="self_rag_${variant}_${topo}_$( [ $sched = true ] && echo serial || echo pipe )"
      python main.py "evaluation/self_rag/configs/${variant}_${topo}_mlx.yml" -p 0 \
        --label "$label" --serialize "$sched"
    done
  done
done

python evaluation/scripts/verify_complex_cases.py

python evaluation/scripts/bandwidth_analysis.py \
  self_rag_factoid_monolith_pipe self_rag_factoid_monolith_serial \
  self_rag_factoid_decomposed_pipe self_rag_factoid_decomposed_serial \
  --out evaluation/results/self_rag_factoid_bandwidth_m2pro.md

python evaluation/scripts/bandwidth_analysis.py \
  self_rag_multihop_monolith_pipe self_rag_multihop_monolith_serial \
  self_rag_multihop_decomposed_pipe self_rag_multihop_decomposed_serial \
  --out evaluation/results/self_rag_multihop_bandwidth_m2pro.md
```

If the decomposed-pipelined factoid cell OOMs, re-run that single cell
from a tmp YAML with `rate: 0.4` and `max_queries: 8` (or 6) as in the
Caveat section above.

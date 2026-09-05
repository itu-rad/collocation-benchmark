# §5.1 — Self-RAG execution strategies (case study)

**Status: being reworked to the agreed §5 plan.** The previous contents of this file (run
commands, arm tables from the "prefill/decode flip" framing) were stale and have been removed.
Authoritative framing: `EXPERIMENTS.md` → *E4 / §5.1*.

## What this experiment is

A **tool-paper case study**, not a findings section. Register is **investigative, not consumer**:
a consumer benchmark outputs a verdict (this is faster); this outputs a *causal account* — where
time and memory go, and which explanation survives isolation. Rankings appear only as the
observation that demands explanation.

**Question.** What does decomposing an agentic pipeline actually change — where do time and memory
go under each execution strategy, what causes the differences (phase mix, lock serialization,
resident weights, retry control flow), and does the causal picture shift across hardware?

## Arms — four execution strategies, differing only in YAML

| | strategy | mechanism |
|---|---|---|
| A | monolithic prompt | one model, grade+answer+check in one pass (8-stage graph) |
| B | shared + locked | `depends_on_id` → one instance, one mutex (13-stage graph) |
| C | per-role copies | independent instances per role |
| D | server, continuous batching | `llm_server` (Ollama) — **never run yet; go/no-go** |

× {factoid, multihop} × {m3pro 18 GB MLX 4-bit, gb10 120 GB HF NF4}. Qwen3.5 4B/9B.

## Exhibits (in order)

1. Auto-generated pipeline graph + YAML-diff strip; stage-reuse count stated once. [C2]
2. Interplay table — quality × latency × throughput × peak memory × power/energy. [C4]
3. Phase breakdown **with a hardware track overlaid** — the unified profiler showing why. [C3]

## Data protocol

- **Latency/quality** — listener-off serial runs; do **not** re-collect these under listeners,
  which would fold an observer cost into the headline table.
  - **gb10**: the existing **R=4** runs in `results/cuda/` are reused (drop r1 → R=3).
  - **m3pro**: collected fresh at **R=6**. The pre-existing runs in `results/mlx/` are **M2 Pro
    data, not m3pro** — see the provenance note below.
- **Power/energy/memory** — from **new R=2 listener-on passes**; the split is stated in methods.
- **Observer cost** — needs no extra collection. `gen_listener_configs.py` builds each `_obs_`
  config as an exact twin of its listener-off counterpart (verified: the two differ only in `name`
  and `listeners`), so every config is run both ways and the on/off comparison is available for
  all of them, not just one paired cell.

  **Measured on both machines** (all 8 configs run both ways; 2026-09-03). The listeners cost
  nothing detectable at this granularity — the shifts are scattered in both directions and sit
  inside run-to-run noise. Reproduce with `analyze_e4.py --pass serial` against `--pass obs`.

  | | listeners | median shift | range |
  |---|---|--:|---|
  | **m3pro** | macmon | **+0.0%** | −0.2% .. +0.2% |
  | **gb10** | dcgmi + top | **−0.7%** | −3.4% .. +1.4% |

  The m3pro figure settles a specific worry in the plan — that macmon logs synchronously and its
  cost had to be bounded. With the batched-logger patch (`0005-macmon-use-batched-logger`) it is
  unmeasurable: no cell moves more than 0.2% in either phase.
- **Throughput** — Poisson cells re-collected under the derived-λ rule (the existing ones are
  R=1/truncated).

### Provenance — the `mlx` runs predate the current Mac

The serial runs tagged `mlx` (last collected 2026-08-23) came from the older **16 GB M2 Pro**, not
from the 18 GB m3pro the paper reports. Three independent checks agree:

1. m3pro had **no `sentence_transformers` installed** and **no `e5-base-v2` cached** — the retriever
   could not have run there, and a smoke run failed on exactly that import.
2. The M2 Pro has both, plus the model in its HuggingFace cache.
3. `CHOREO_FINDINGS.md`'s own setup table says it outright: *"mlx = M2 Pro (4-bit OptiQ)"*.

**The three Apple machines, verified 2026-09-02** (`sysctl machdep.cpu.brand_string`):

| token | host | chip | RAM | role |
|---|---|---|---|---|
| `m2pro` / `mlx` | mac623807 | Apple M2 Pro | 16 GB | staging; produced the superseded `mlx` runs |
| `m3pro` | mac624090 (`itu-mac`) | Apple M3 Pro | **18 GB** | the Apple machine the paper reports |
| `gb10` | spark-cc0d (`babyxena`) | NVIDIA GB10 | 120 GB | the accelerator machine |

E1, E2 and E3 all carry m3pro results, so m3pro is the paper's Apple machine and the `mlx` E4
runs are the outlier — not the other way round.

Reusing them would have paired the M2 Pro's latencies with a memory column describing a different machine —
and the memory-budget question (what a 5× larger budget buys) is one of the section's exhibits, so
the error would have landed directly in a headline claim. **Every `mlx` number in
`CHOREO_FINDINGS.md` is therefore M2 Pro data**; the `cuda` numbers there are genuine gb10.

## Layout

    configs/                 the arms (YAML-only differences)
    results/cuda/            R=4 serial runs (gb10), reusable for latency/quality
    results/mlx/             M2 Pro runs -- NOT m3pro; superseded by results/m3pro/
    results/m3pro/           R=6 serial runs collected on the current Mac
    results/quantest/        bf16 datum from the retired cost-law work; orphaned, kept as data
    judge/                   Haiku LLM-judge inputs + verdicts (quality column)
    analyze_e4.py            analyzer (CSV path carries §5.1; spans migration is not blocking)
    gen_serialized_configs.py
    collect_e4.sh            the two passes (serial = latency/quality, obs = counters)
    collect.sh               superseded by collect_e4.sh
    stage_latency.py · retry_analysis.py · retry_tail.py   to be folded into the analyzer

## Carried over from the retired findings document

`CHOREO_FINDINGS.md` was removed (superseded framing, working title in the name, and every `mlx`
number in it was M2 Pro data). Two things in it are method, not superseded results, and are kept
here.

### Why the retriever is `e5-base-v2` at top_k=5

The first pass used ChromaDB's default `all-MiniLM-L6-v2` at top_k=3, which capped accuracy by
mis-ranking supporting passages. It was rebuilt with **`e5-base-v2` at top_k=5** — the embedder
MLPerf's `e2e-rag` uses — and everything re-run. Retrieval-hit rose 0.62 → 0.705 (factoid) and
0.43 → 0.50 (multihop); judge accuracy rose 8–11 pp on factoid.

This matters for how the strategies are read: splitting questions by whether the gold answer was
retrieved at all, a 4B extracts as well as the 9B when retrieval succeeds and neither recovers when
it fails (≈88% of shared factoid failures are retrieval misses). So the quality column is
**retrieval-bound**, and the strategies are quality-comparable by construction rather than by luck —
which is why the section can compare them on latency, memory and power without a quality confound.
The conclusion held at both retriever strengths, so it is a property of the task, not of a weak
retriever.

### Greedy decoding is deterministic within a machine, not across machines

Measured 2026-09-05, same model, same 4-bit quantisation, same config, greedy:

| | answers byte-identical |
|---|---|
| m3pro, repetition vs repetition | **30/30** in every cell tested |
| m3pro vs M2 Pro, same cell | **149/180 (31 differ)** |

Per cell across machines: factoid 29/30, 29/30, 28/30; multihop 27/30, 20/30, and
**16/30 for the 9B monolith** — divergence grows with generation length, which is what you would
expect if it comes from accumulated floating-point differences rather than sampling.

Two consequences:

1. **"R=1 is exact for quality" holds per machine and only per machine.** Within m3pro the answers
   are bit-identical across repetitions, so one repetition suffices — but the claim cannot be
   stated machine-independently.
2. **The quality column must be judged per machine.** Carrying the M2 Pro verdicts over to m3pro
   would mis-score 17% of answers overall and 47% in the worst cell. This is an independent reason
   the m3pro half had to be re-collected, separate from the memory-budget argument: the quality
   numbers would have been wrong too, and silently so, since the two sets of answers look alike.

The divergences are not cosmetic — one flips a wrong answer to a right one (*"George Meade"* →
*"George McClellan"* for the general at Antietam).

### The quality column: same questions, both machines

Scored 2026-09-05 from the counter-pass outputs, which cover the same 30 questions on both
machines. Judge = Claude Haiku via the CLI, one judge per cell.

| task | strategy | m3pro | gb10 |
|---|---|--:|--:|
| factoid | monolith (9B) | 0.867 | 0.933 |
| factoid | monolith_4b | 0.933 | 0.900 |
| factoid | decomposed | 0.867 | 0.867 |
| factoid | decomposed_shared | 0.867 | 0.867 |
| multihop | monolith (9B) | 0.433 | 0.300 |
| multihop | monolith_4b | 0.333 | 0.433 |
| multihop | decomposed | 0.433 | 0.433 |
| multihop | decomposed_shared | 0.433 | 0.433 |

The strategies are quality-comparable, which is what lets the section compare them on latency,
memory and power without a quality confound. The 9B never pulls clear of the 4B — consistent with
the retrieval-bound ceiling above.

**Why this was re-derived.** gb10's published quality came from a dedicated `_quality_` collection
of 120 questions, whose configs lived in the removed `evaluation/collect/` tree and no longer exist
— so that column was **not reproducible from this repo**, and it rested on a different question set
from m3pro's. Both machines are now scored from committed configs on the same 30 questions. The
older 120-question `cuda_*` verdicts are kept as `cuda_*` and are not the reported column.

### Why quality is scored by LLM judge, not exact match

Exact match materially mis-ranks these strategies (it penalises correct answers phrased differently),
so quality is scored three ways — exact match, token-F1, and a **Haiku LLM judge** for semantic
equivalence. The judge is the one the section reports; `run_judge.py` reproduces it, and its
`score` subcommand re-derives every cell from the stored verdicts with no API calls. Greedy decoding
makes answers byte-identical across repetitions, so R=1 is exact for quality.

## Not doing

Predictive cost law · unit-of-measurement critique (that is §4/E3's register) · knob sweeps
(`top_k`, `max_retries`, `max_new_tokens` are held fixed by the pre-registered R-PRECEDENT rule) ·
vLLM · MIG beyond a one-line verification.

## Caveat to state in the paper

This is **not** Asai's reflection-token Self-RAG, which amortizes critique into one decoding pass;
ours issues separate LLM calls.

## Known objections, and where each is answered

Carried over from the three-round mock review of the earlier draft of this section. The
framing has changed but these objections still land, and each must stay answered:

- **"This is not Self-RAG."** Correct. Asai's reflection-token Self-RAG amortizes critique into
  one decoding pass; ours issues **separate LLM calls** per role — the pattern deployed agentic
  frameworks use (cf. CRAG, Adaptive-RAG). Say it plainly and early.
- **"Your grader cost is an equal-size-grader artifact."** Fair. The auxiliary calls are
  call-count-driven with same-size graders; lightweight graders, prefix caching or continuous
  batching would shrink them. Scope the claim to the configuration measured.
- **"You compare arms on latency without controlling quality."** Fatal if unreported, trivial if
  reported. Quality goes beside **every** latency or throughput ranking — never a table without
  it.
- **"Retry-vs-correctness is selection-confounded."** It is: the graders trigger retries
  precisely on hard queries, so a retried-vs-never-retried score gap is descriptive of
  selection, not causal evidence of waste. Report it as descriptive.
- **"The scheduler gap is mischaracterised."** Per-request engines treat a retry as a fresh
  request; the invisibility lives at the **orchestration layer**, not in the engine. State it
  that way.
- **"n=1 workload, n=2 devices."** Answered by claim discipline: an existence proof that an
  abstraction fails needs one counterexample, not a population. Write every claim that way.

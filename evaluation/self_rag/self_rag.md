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

× {factoid, multihop} × {m3pro 24 GB MLX 4-bit, gb10 120 GB HF NF4}. Qwen3.5 4B/9B.

## Exhibits (in order)

1. Auto-generated pipeline graph + YAML-diff strip; stage-reuse count stated once. [C2]
2. Interplay table — quality × latency × throughput × peak memory × power/energy. [C4]
3. Phase breakdown **with a hardware track overlaid** — the unified profiler showing why. [C3]

## Data protocol

- **Latency/quality** — from the existing **R=4 listener-off serial runs** in `results/{cuda,mlx}/`
  (drop r1 → R=3). Do **not** re-collect these under listeners.
- **Power/energy/memory** — from **new R=2 listener-on passes**; one paired on/off cell bounds the
  observer cost. The split is stated in methods.
- **Throughput** — Poisson cells re-collected under the derived-λ rule (the existing ones are
  R=1/truncated).

## Layout

    configs/                 the arms (YAML-only differences)
    results/{cuda,mlx}/      R=4 serial runs, reusable for latency/quality
    results/quantest/        bf16 datum from the retired cost-law work; orphaned, kept as data
    judge/                   Haiku LLM-judge inputs + verdicts (quality column)
    analyze_e4.py            analyzer (CSV path carries §5.1; spans migration is not blocking)
    gen_serialized_configs.py
    collect.sh               to be replaced by collect_e4.sh (collect_e3.sh pattern)
    stage_latency.py · retry_analysis.py · retry_tail.py   to be folded into the analyzer

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

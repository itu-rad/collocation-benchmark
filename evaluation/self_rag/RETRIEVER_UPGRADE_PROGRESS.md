# Self-RAG retriever upgrade + full re-run — progress checkpoint

**Goal:** the Self-RAG accuracy ceiling was retrieval-bound by a weak retriever (ChromaDB default
`all-MiniLM-L6-v2`, top_k=3), not the LLM. Strengthen the retriever and re-run the full quality matrix
to see whether it raises the ceiling / changes the arm story.

## Decision (locked)
**Embedder = `intfloat/e5-base-v2`** (what MLPerf `e2e-rag` uses), **top_k = 5**.
Offline ablation (entity gold-recall@k, substring lower bound):

| | @3 | @5 | @10 |
|---|---|---|---|
| factoid MiniLM (old) | 0.590 | 0.639 | 0.672 |
| factoid **e5** | 0.672 | **0.705** | 0.721 |
| multihop MiniLM (old) | 0.377 | 0.425 | 0.481 |
| multihop **e5** | 0.458 | **0.500** | 0.583 |

## Done
- `ChromaRetriever` now takes `embedding_model` / `query_prefix` / `passage_prefix`; embeddings are
  **precomputed** so e5's asymmetric `query:`/`passage:` prefixes apply (ChromaDB's built-in EF can't).
  All 26–27 self_rag configs switched to e5 + top_k=5.
- **Committed on both boxes** (no Claude attribution, no push): M2 + babyxena. On babyxena the 112
  in-progress overhead-result CSVs were **parked in `git stash@{0}`** (not mine) to get a clean tree;
  `git stash pop` to restore them.
- MiniLM baseline preserved: `evaluation/self_rag/_baseline_miniLM/` (outputs + judge verdicts + report
  snapshot).
- **Smoke validated both boxes** (factoid_monolith_4b): mlx retrieval-hit **0.607→0.705**, EM
  **0.575→0.675** — stronger retriever lifts retrieval *and* accuracy, as predicted.

## In progress
- **Full re-run: 8 quality cells/box, `run_collection.py --only 'e4_*_quality' --force`.**
  cuda on GB10 (`~/collocation-benchmark/scratchpad/rerun_cuda.log`), mlx on M2
  (`scratchpad/rerun_mlx.log`). Logs to MLflow exp 138 with power listeners.
  Completion waiter: background task `bbc2roux8`.

## Pending (staged, run in order once re-run completes)
1. `scratchpad/judge_extract.py` → re-extract QA triples from new outputs.
2. Judge fleet: Workflow `llm-judge-selfrag` (Haiku, 12 cells) → verdicts; validate overturn=0, re-judge
   any corrupt cell.
3. Recompute: accuracy (EM/F1/judge), retrieval hit/miss + coverage-vs-ranking, tokens/correct,
   latency/throughput, Finding-1 prefill (top_k 3→5 raises prefill), Finding-2 efficiency.
4. `scratchpad/pull_energy.py <cell>` (validated) → new J/correct for cuda cells.
5. Update `evaluation/self_rag/CHOREO_FINDINGS.md`: §0 accuracy, retrieval-ceiling section, Findings 1–2,
   synthesis; add the retriever-upgrade note + before/after.

## RESULTS (e5 re-run complete — cuda 8/8, mlx 8/8, 0 failures; judge overturn=0 all cells)

**Answer to the key question: the ceiling ROSE but the arms stay quality-comparable — model size still
does not win.** Even on retrieval-*hit* questions the 9B ties the 4B (cuda factoid EM 0.72 vs 0.77;
multihop 0.47 vs 0.45). Retrieval, not capacity, remains the bottleneck — now at a higher level. This
makes the finding robust to the "your retriever was weak" objection.

### Retrieval entity hit-rate (MiniLM → e5)
cuda factoid 0.623→0.705 · cuda multihop 0.434→0.500 · mlx factoid 0.607→0.705 · mlx multihop 0.443→0.500

### Accuracy — judge (MiniLM → e5)
| cell | monolith(9B) | monolith_4b | decomposed |
|---|---|---|---|
| cuda factoid | 0.725→**0.808** | 0.750→**0.842** | 0.750→**0.842** |
| cuda multihop | 0.308→0.342 | 0.367→0.367 | 0.350→**0.375** |
| mlx factoid | 0.683→**0.792** | 0.692→**0.842** | 0.733→**0.808** |
| mlx multihop | 0.300→0.358 | 0.333→**0.392** | 0.333→0.383 |
EM also up across the board (cuda factoid 0.45/0.61/0.65 → 0.49/0.67/0.68).

### Latency / throughput (e5)
- cuda factoid p50: mono 2.7s / 4b 1.7s / dec 1.9s; thr 0.329 / 0.481 / 0.474 q/s.
- **Device split sharpened:** mlx factoid decomposed is now SLOW (p50 9.0s, p90 20.1s) vs 4b (3.3s) —
  top_k=5 means decomposed re-encodes 5 docs × 3 roles, and that prefill tax is brutal on the M2.

### Finding 2 — efficiency per JUDGE-correct (cuda), MiniLM → e5
| | J/corr | tok/corr |
|---|---|---|
| factoid mono(9B) | 259→**198** | 151→49 |
| factoid 4b | 138→**124** | 54→39 |
| factoid decomposed | 145→**160** | 28→**22** |
| multihop decomposed | 449→**435** | 66→**62** |

**The token-vs-energy divergence got SHARPER and is the headline:** on cuda factoid, decomposed uses the
FEWEST tokens/correct (22 vs 39 vs 49) yet MORE energy than the same-size 4B monolith (160 vs 124 J) —
because top_k=5 makes its 3× doc re-encoding (prefill) fully outweigh the decode it saves. A token cost
proxy says decomposed is ~1.8× cheaper; the joules say it's ~30% more expensive. Only phase-split +
power reveals it. (Structure energy now clearly favors the plain 4B monolith on factoid; decomposed's
edge survives only on multihop where it generates far fewer tokens.)

## COMPLETE — report fully recomputed on e5, all findings, both devices
- ✅ F3 recomputed on e5, both devices: composition (86–100% relevance=no) + rescue + early-exit.
  cuda joined by **question-text match** (query_id doesn't join on cuda). **Key correction:** cuda
  factoid rescue is **70–71%** (early-exit costs 12.5pp) — the retry loop is genuinely valuable there;
  the mlx-only view ("cheap to drop") was misleading and is fixed.
- ✅ mlx energy pulled (`system/macmon - All Power`, total-SoC): the token-vs-energy inversion is 2× on
  the M2 (decomposed 319 vs 156 J/correct vs the 4B monolith on factoid).
- ✅ All 16 cells done; `CHOREO_FINDINGS.md` consistent on e5 throughout; **left uncommitted for review.**

## Only genuinely-open items (future work / framing, not analysis gaps)
- F3's *dynamic* early-exit discriminator (answer stationarity) — proposed, buildable (per-attempt data
  now joined), not yet built.
- F1's 62% hoist is a priced ceiling, not an implemented optimization.
- Prior-art positioning (PromptCache/APC, FrugalGPT, CRAG/Adaptive-RAG).

## DONE (2026-07-23)
- ✅ Retriever upgraded to e5-base-v2 + top_k=5, committed both boxes.
- ✅ Full re-run: cuda 8/8, mlx 7/8 (+1 perf dup finishing), 0 failures.
- ✅ Judge re-score all 12 cells, overturn=0, verdicts persisted to `evaluation/self_rag/judge/`.
- ✅ Recompute: accuracy (EM/F1/judge), retrieval hit/miss (§R), prefill (F1), efficiency (F2), latency.
- ✅ `CHOREO_FINDINGS.md` updated throughout to e5 + before/after framing + new §R section.
  **Report is NOT committed** (untracked) — left for user review (F3 still pending).

## Headline changes the re-run produced
1. Ceiling rose (+8–11pp judge on factoid) but arms stay comparable — 9B still never wins → finding is
   robust to "weak retriever" (§R). This is the strongest version of the retrieval-bound story.
2. **decomposed is no longer a latency win** (top_k=5 prefill tax); monolith_4b is now fastest. Choreo's
   own measurement overturned the earlier conclusion once the toy-retriever confound was removed.
3. **Token-vs-energy divergence sharpened to an inversion**: decomposed uses ½ the tokens yet +30%
   energy vs the 4B monolith on factoid. Strongest illustration of "measure joules, not tokens."

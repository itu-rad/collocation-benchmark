# LLM-as-judge semantic scoring — Self-RAG quality cells

Exact-match materially mis-ranks the Self-RAG arms: it penalizes verbose answers (a 9B answering a
yes/no question with `"The Celsius crater on the Moon is named after him"` instead of `"yes"` scores
EM=0 though correct) and cannot read through OCR-corrupted gold in `rag-mini-wikipedia`
(e.g. `"MassachuS08_setts"`). We add a semantic LLM judge alongside EM/F1.

## Method
- **Judge model:** Claude Haiku (`claude-haiku-4-5`), one judge per cell, `effort=medium`.
- **Unit:** each `(question, gold[], candidate answer)` triple from a quality cell's
  `*_outputs.jsonl` (`_best_answer` extracts the candidate, matching the EM scorer).
- **Rubric:** judge *factual equivalence* only — ignore form/verbosity/case/punctuation; credit
  correct yes/no affirmation or denial without the literal token; credit name variants, entailment,
  and the asked field even if gold over-specifies; read through OCR typos; mark empty/refusal/wrong
  fact incorrect. Judge each item independently. (Full prompt in the workflow script /
  `verify_*` transcripts.)
- **Anchoring guard:** judges see `*_input.json` (no EM/F1 fields), not the scored files.

## Validity checks (why this is trustworthy despite a non-deterministic judge)
- **One-directional:** across all 12 cells the judge *only rescues EM false-negatives* — **zero clean
  EM matches overturned**. That is the expected signature if EM under-counts and the judge corrects it.
- **Self-caught failure:** one cell (`mlx_multihop_decomposed`) initially failed 11 exact-string
  matches (`"yes"`→wrong) — flagged by the overturn≠0 check and re-judged clean.
- **Hand audit:** the 34 rescues on `cuda_factoid_monolith` were inspected — ~33/34 legitimate,
  ~1 borderline.
- **Run-to-run noise ≈ ±1–2 / 120 (~1%)** (two independent passes on the 9B cell gave 88 vs 87).
- Deterministic F1 is retained beside the judge as the reproducibility anchor.

## Files
- `{dev}_{task}_{arm}_input.json` — anchoring-free judge inputs (`i,q,gold,ans`).
- `verdicts_{dev}_{task}_{arm}.json` — `[{i, correct}]`, 120 items/cell.
- `manifest.json` — cell list + EM.
- Regenerate inputs: extraction snippet in the session transcript; re-judge via the
  `llm-judge-selfrag` workflow.

## Headline result
Under the judge the EM spread collapses — arms are quality-comparable (cuda factoid EM 0.45/0.61/0.65
→ judge 0.725/0.750/0.750). The 9B is **not** the weak arm; its EM deficit is an answer-format
artifact on yes/no questions (on entity questions it is the *most* accurate). Corrects §0 (baseline)
and Finding 2 (per-correct efficiency: size energy gap 2.5×→1.8×; structure energy break-even while
tokens/correct still ~2× apart — the token-vs-joule divergence).

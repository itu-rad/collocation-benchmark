"""Shared answer-style instruction for every answer-producing Self-RAG arm.

EM/F1 (SQuAD-style, evaluation/scripts/score_quality.py) assume short-span
answers; free-form sentences score EM=0 by construction and dilute F1
(measured 2026-07-13: EM 0.000 / F1 0.16-0.19 on the smoke arms while
containment read 0.7-0.8). Every arm of a comparison must therefore request
the same short-answer style — defined ONCE here so the monolith and the
decomposed generator can never drift apart (the arms' prompts differ in
*task structure*, never in answer style).

Changing this string invalidates comparability with all previously collected
quality data; it landed before the post-`query_id`-fix re-collection pass.
"""

SHORT_ANSWER_STYLE = (
    "Answer format (STRICT): reply with only the shortest exact answer — a "
    "name, entity, phrase, number, or 'yes'/'no' — copied from the documents "
    "where possible. Do NOT write a full sentence. No explanations, no "
    "restating the question, no punctuation beyond the answer itself. "
    "Examples: 'yes'; 'Abraham Lincoln'; '1862'; 'the Legal Tender Act'."
)

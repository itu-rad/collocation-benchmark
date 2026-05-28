"""Semantic verification of the VQA and Self-RAG complex case pipelines.

Reads `evaluation/results/<pipeline>_outputs.jsonl` produced by the
TerminalCapture sidecar stage, applies per-pipeline pass/fail criteria,
and writes a Markdown report to
`evaluation/results/verification_report.md`.

Usage:
    python evaluation/scripts/verify_complex_cases.py            # all pipelines
    python evaluation/scripts/verify_complex_cases.py self_rag_monolith
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


RESULTS_DIR = Path("evaluation/results")
REPORT_PATH = RESULTS_DIR / "verification_report.md"

# Pipeline-name → jsonl basename (matches main.py's pipeline-name-to-CSV mangling
# and the TerminalCapture stage's _outputs.jsonl naming).
PIPELINES = {
    "vqa_mps_monolith":         {"label": "VQA Mapping A (MPS monolith)",      "family": "vqa"},
    "vqa_heterogeneous_split":  {"label": "VQA Mapping B (heterogeneous CoreML)", "family": "vqa"},
    # Smoke-test labels (single run per topology):
    "self_rag_monolith":        {"label": "Self-RAG Monolith",                 "family": "self_rag"},
    "self_rag_decomposed":      {"label": "Self-RAG Decomposed",               "family": "self_rag"},
    # 2x2 cell labels for the factoid (rag-mini-wikipedia) and multi-hop (HotpotQA)
    # task families. Each topology gets two schedule cells (pipelined / serial).
    "self_rag_factoid_monolith_pipe":      {"label": "Self-RAG Factoid Monolith (pipelined)",        "family": "self_rag"},
    "self_rag_factoid_monolith_serial":    {"label": "Self-RAG Factoid Monolith (serial)",           "family": "self_rag"},
    "self_rag_factoid_decomposed_pipe":    {"label": "Self-RAG Factoid Decomposed (pipelined)",      "family": "self_rag"},
    "self_rag_factoid_decomposed_serial":  {"label": "Self-RAG Factoid Decomposed (serial)",         "family": "self_rag"},
    "self_rag_multihop_monolith_pipe":     {"label": "Self-RAG Multi-hop Monolith (pipelined)",      "family": "self_rag"},
    "self_rag_multihop_monolith_serial":   {"label": "Self-RAG Multi-hop Monolith (serial)",         "family": "self_rag"},
    "self_rag_multihop_decomposed_pipe":   {"label": "Self-RAG Multi-hop Decomposed (pipelined)",    "family": "self_rag"},
    "self_rag_multihop_decomposed_serial": {"label": "Self-RAG Multi-hop Decomposed (serial)",       "family": "self_rag"},
}

# Hard-failure markers placed in final_data by routers when retries exhaust.
ERROR_MARKERS = (
    "Error: No more retries left",
    "Error: no satisfactory answer after retries",
)


@dataclass
class QueryEval:
    query_id: str
    question: str | None
    golden_answers: list[str] = field(default_factory=list)
    retrieved_documents: list[str] = field(default_factory=list)
    generated_answer: str | None = None
    final_data: str | None = None

    @property
    def best_answer(self) -> str | None:
        """generated_answer (set by routers/halluc grader) wins; fall back to final_data."""
        for candidate in (self.generated_answer, self.final_data):
            if candidate is None:
                continue
            if isinstance(candidate, list):
                candidate = " ".join(str(x) for x in candidate)
            candidate = str(candidate).strip()
            if not candidate:
                continue
            if any(marker in candidate for marker in ERROR_MARKERS):
                continue
            return candidate
        return None


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _golden_hit(answer: str, golden: Iterable[str]) -> bool:
    """True iff any normalized golden answer appears in the normalized generated answer."""
    ans = _normalize(answer)
    if not ans:
        return False
    for g in golden:
        if not g:
            continue
        gn = _normalize(str(g))
        if gn and gn in ans:
            return True
    return False


def _load_jsonl(path: Path) -> list[QueryEval]:
    out: list[QueryEval] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(QueryEval(
            query_id=str(r.get("query_id", "")),
            question=r.get("question"),
            golden_answers=list(r.get("golden_answers") or []),
            retrieved_documents=list(r.get("retrieved_documents") or []),
            generated_answer=r.get("generated_answer"),
            final_data=r.get("final_data"),
        ))
    return out


def _summarize(label: str, queries: list[QueryEval]) -> dict:
    if not queries:
        return {
            "label": label, "n": 0, "answered": 0, "hit": 0,
            "avg_len": 0, "error_rate": 1.0,
            "rows": [],
        }

    rows = []
    answered = 0
    hits = 0
    lens = []
    errors = 0

    for q in queries:
        ans = q.best_answer
        is_error = ans is None
        if is_error:
            errors += 1
        else:
            answered += 1
            lens.append(len(ans))
        hit = _golden_hit(ans, q.golden_answers) if ans else False
        if hit:
            hits += 1
        rows.append({
            "query_id": q.query_id,
            "question": q.question,
            "golden": q.golden_answers,
            "answer": ans,
            "n_docs": len(q.retrieved_documents),
            "first_doc": (q.retrieved_documents[0][:120] + "...") if q.retrieved_documents else None,
            "hit": hit,
            "is_error": is_error,
        })

    n = len(queries)
    return {
        "label": label,
        "n": n,
        "answered": answered,
        "hit": hits,
        "avg_len": int(sum(lens) / len(lens)) if lens else 0,
        "error_rate": errors / n,
        "rows": rows,
    }


def _parity(rows_a: list[dict], rows_b: list[dict]) -> tuple[int, int, float]:
    """Return (compared, matched, avg_overlap_ratio).

    Match on question (most reliable). For each shared question, compute the
    Jaccard of normalized whitespace-tokenized answers.
    """
    by_q_a = {r["question"]: r for r in rows_a if r["question"] and r["answer"]}
    by_q_b = {r["question"]: r for r in rows_b if r["question"] and r["answer"]}
    shared = sorted(set(by_q_a) & set(by_q_b))
    matched = 0
    ratios = []
    for q in shared:
        toks_a = set(_normalize(by_q_a[q]["answer"]).split())
        toks_b = set(_normalize(by_q_b[q]["answer"]).split())
        if not toks_a or not toks_b:
            continue
        inter = len(toks_a & toks_b)
        union = len(toks_a | toks_b)
        r = inter / union if union else 0.0
        ratios.append(r)
        if r >= 0.5:
            matched += 1
    return len(shared), matched, (sum(ratios) / len(ratios) if ratios else 0.0)


def _render_pipeline_section(summary: dict) -> str:
    lines = [f"### {summary['label']}", ""]
    n = summary["n"]
    if n == 0:
        lines.append("_No output file found — pipeline has not been run yet, or capture failed._")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"- Queries: **{n}** | answered: **{summary['answered']}** "
        f"({summary['answered']/n:.0%}) | "
        f"golden-overlap hits: **{summary['hit']}** "
        f"({summary['hit']/n:.0%}) | "
        f"avg answer length: {summary['avg_len']} chars | "
        f"error rate: {summary['error_rate']:.0%}"
    )
    lines.append("")
    lines.append("| # | Question | Golden | Answer | n_docs | hit |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(summary["rows"], 1):
        q = (r["question"] or "")[:80].replace("|", "\\|")
        g = ", ".join(str(x) for x in (r["golden"] or []))[:80].replace("|", "\\|")
        a = (r["answer"] or "(no answer)")[:120].replace("|", "\\|").replace("\n", " ")
        h = "✓" if r["hit"] else ("error" if r["is_error"] else "—")
        lines.append(f"| {i} | {q} | {g} | {a} | {r['n_docs']} | {h} |")
    lines.append("")
    return "\n".join(lines)


def _bandwidth_check(summary: dict, pipeline_key: str) -> list[str]:
    """Heuristic pipeline-essence checks (no behavior verification, just shape)."""
    notes = []
    rows = summary.get("rows") or []
    if not rows:
        return notes
    n_docs_seen = [r["n_docs"] for r in rows]
    if all(n == 0 for n in n_docs_seen):
        notes.append("⚠ retrieval returned 0 docs for every query — corpus may be empty or text_column wrong")
    elif min(n_docs_seen) == 0:
        notes.append(f"⚠ retrieval returned 0 docs for {sum(1 for n in n_docs_seen if n==0)}/{len(rows)} queries")
    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pipelines", nargs="*",
        help="pipeline keys to verify (default: all). "
             "Choices: " + ", ".join(PIPELINES),
    )
    args = parser.parse_args()

    keys = args.pipelines or list(PIPELINES)
    unknown = [k for k in keys if k not in PIPELINES]
    if unknown:
        sys.exit(f"unknown pipeline key(s): {unknown}; choices: {list(PIPELINES)}")

    summaries: dict[str, dict] = {}
    for k in keys:
        meta = PIPELINES[k]
        jsonl = RESULTS_DIR / f"{k}_outputs.jsonl"
        queries = _load_jsonl(jsonl)
        summaries[k] = _summarize(meta["label"], queries)
        summaries[k]["_family"] = meta["family"]
        summaries[k]["_jsonl"] = str(jsonl)

    # Build report.
    out = ["# Verification report — complex case pipelines", ""]
    out.append("Generated by `evaluation/scripts/verify_complex_cases.py`.")
    out.append("Reads `evaluation/results/<pipeline>_outputs.jsonl` written by `stages.TerminalCapture`.")
    out.append("")

    out.append("## Per-pipeline results")
    out.append("")
    for k in keys:
        out.append(_render_pipeline_section(summaries[k]))
        notes = _bandwidth_check(summaries[k], k)
        if notes:
            for n in notes:
                out.append(f"- {n}")
            out.append("")

    # Cross-pipeline parity: A vs B for VQA, monolith vs decomposed for Self-RAG.
    out.append("## Cross-pipeline parity")
    out.append("")
    if {"vqa_mps_monolith", "vqa_heterogeneous_split"}.issubset(summaries):
        compared, matched, avg = _parity(
            summaries["vqa_mps_monolith"]["rows"],
            summaries["vqa_heterogeneous_split"]["rows"],
        )
        out.append(
            f"- **VQA A vs B** (same CLIP weights, only encoder backend differs — expect high parity):  "
            f"compared {compared} shared questions; "
            f"{matched} matched (Jaccard ≥ 0.5); avg Jaccard = {avg:.2f}"
        )
    if {"self_rag_monolith", "self_rag_decomposed"}.issubset(summaries):
        compared, matched, avg = _parity(
            summaries["self_rag_monolith"]["rows"],
            summaries["self_rag_decomposed"]["rows"],
        )
        out.append(
            f"- **Self-RAG monolith vs decomposed** (different decomposition, different model sizes — moderate parity expected):  "
            f"compared {compared} shared questions; "
            f"{matched} matched (Jaccard ≥ 0.5); avg Jaccard = {avg:.2f}"
        )
    out.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")

    # Brief stdout summary.
    for k, s in summaries.items():
        n = s["n"]
        if n:
            print(f"  {k:30s} n={n} answered={s['answered']} hits={s['hit']} err_rate={s['error_rate']:.0%}")
        else:
            print(f"  {k:30s} (no output file)")


if __name__ == "__main__":
    main()

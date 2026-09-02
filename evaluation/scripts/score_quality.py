"""Answer-quality scoring (EM / token-F1) for the RAG case studies.

This is the accuracy-of-record scorer the methodology promises: **exact match
(EM)** and **token-level F1** with SQuAD-style normalization (lowercase, strip
punctuation and articles, collapse whitespace), computed over the
`evaluation/results/<label>_outputs.jsonl` sidecars written by
`stages.TerminalCapture`. The substring-containment rate of
`verify_complex_cases.py` is kept only as a *secondary* signal and is always
reported next to mean answer length (containment is gameable by verbosity).

Statistics follow the paper's rules (experimental_setup.tex §Statistics):
Wilson 95% intervals on pooled rate metrics, plus — when an arm has R > 1 run
files — the raw per-run values and a hierarchical (cluster) bootstrap CI that
resamples runs first, then queries within each resampled run (10^4 resamples,
fixed seed). With few runs the interval is indicative; the raw run values are
always printed beside it.

Usage:
    # each positional is ARM=GLOB; the glob matches *_outputs.jsonl basenames
    # (without the suffix) under evaluation/results/, one match = one run
    python evaluation/scripts/score_quality.py \
        monolith='self_rag_factoid_monolith_r*' \
        decomposed='self_rag_factoid_decomposed_r*'

    # bare labels work too (arm name = label, single run)
    python evaluation/scripts/score_quality.py self_rag_monolith self_rag_decomposed

Writes `evaluation/results/quality_report.md` (+ `.json`) and prints a summary.
Cross-arm parity (token-Jaccard on shared questions) is reported for every
arm pair.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Quality sidecars live beside the experiment that produced them (E2/E3/E4
# pattern); evaluation/results/ is only main.py's staging dir. Override with
# --results-dir for any other experiment.
RESULTS_DIR = Path("evaluation/self_rag/results")
REPORT_MD = RESULTS_DIR / "quality_report.md"
REPORT_JSON = RESULTS_DIR / "quality_report.json"

N_BOOT = 10_000
BOOT_SEED = 1234
Z95 = 1.959963984540054

# Hard-failure markers placed in final_data by routers when retries exhaust
# (kept in sync with verify_complex_cases.py).
ERROR_MARKERS = (
    "Error: No more retries left",
    "Error: no satisfactory answer after retries",
)

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalization and per-answer metrics (SQuAD-style)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def exact_match(answer: str, goldens: list[str]) -> int:
    """1 iff the normalized answer equals any normalized golden answer."""
    ans = normalize_answer(answer)
    return int(any(ans == normalize_answer(str(g)) for g in goldens if g))


def token_f1(answer: str, goldens: list[str]) -> float:
    """Max over goldens of F1 between token *bags* (multisets)."""
    ans_toks = normalize_answer(answer).split()
    best = 0.0
    for g in goldens:
        if not g:
            continue
        gold_toks = normalize_answer(str(g)).split()
        if not ans_toks or not gold_toks:
            # SQuAD convention: if either side is empty, F1 = 1 iff both empty.
            best = max(best, float(ans_toks == gold_toks))
            continue
        common = Counter(ans_toks) & Counter(gold_toks)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(ans_toks)
        recall = n_common / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def containment(answer: str, goldens: list[str]) -> int:
    """Secondary signal: 1 iff any normalized golden is a substring of the
    normalized answer. Gameable by verbosity — never the accuracy of record."""
    ans = normalize_answer(answer)
    if not ans:
        return 0
    return int(any(
        normalize_answer(str(g)) and normalize_answer(str(g)) in ans
        for g in goldens if g
    ))


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cluster_bootstrap_ci(runs: list[list[float]],
                         n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> tuple[float, float]:
    """Hierarchical bootstrap CI for the pooled mean of per-query values.

    Resample runs with replacement, then queries within each resampled run,
    pool, take the mean; percentile 95% interval over n_boot replicates.
    """
    runs = [r for r in runs if r]
    if not runs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        pooled: list[float] = []
        for _ in range(len(runs)):
            run = runs[rng.randrange(len(runs))]
            pooled.extend(run[rng.randrange(len(run))] for _ in range(len(run)))
        stats.append(sum(pooled) / len(pooled))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(int(0.975 * n_boot), n_boot - 1)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Loading (record semantics shared with verify_complex_cases.py)
# ---------------------------------------------------------------------------

@dataclass
class Record:
    question: str | None
    goldens: list[str] = field(default_factory=list)
    answer: str | None = None  # None = unanswered/error


def _best_answer(r: dict) -> str | None:
    for candidate in (r.get("generated_answer"), r.get("final_data")):
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


def load_run(path: Path) -> list[Record]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(Record(
            question=r.get("question"),
            goldens=[str(g) for g in (r.get("golden_answers") or [])],
            answer=_best_answer(r),
        ))
    return records


def resolve_arms(specs: list[str]) -> dict[str, list[Path]]:
    """ARM=GLOB or bare label -> ordered run files."""
    available = sorted(RESULTS_DIR.glob("*_outputs.jsonl"))
    basenames = {p.name[:-len("_outputs.jsonl")]: p for p in available}
    arms: dict[str, list[Path]] = {}
    for spec in specs:
        name, _, pattern = spec.partition("=")
        pattern = pattern or name
        matches = sorted(b for b in basenames if fnmatch.fnmatch(b, pattern))
        if not matches:
            sys.exit(f"arm '{name}': no *_outputs.jsonl in {RESULTS_DIR} matches '{pattern}' "
                     f"(available: {', '.join(basenames) or 'none'})")
        arms[name] = [basenames[b] for b in matches]
    return arms


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_arm(name: str, run_paths: list[Path]) -> dict:
    runs = [load_run(p) for p in run_paths]
    per_run = []
    # Per-query metric vectors per run (unanswered scores 0 on every metric,
    # so rates are over ALL queries, not just answered ones).
    em_runs, f1_runs, cont_runs, ans_runs = [], [], [], []
    lens = []
    for path, records in zip(run_paths, runs):
        em_v = [float(exact_match(r.answer, r.goldens)) if r.answer else 0.0 for r in records]
        f1_v = [token_f1(r.answer, r.goldens) if r.answer else 0.0 for r in records]
        ct_v = [float(containment(r.answer, r.goldens)) if r.answer else 0.0 for r in records]
        an_v = [1.0 if r.answer else 0.0 for r in records]
        em_runs.append(em_v); f1_runs.append(f1_v)
        cont_runs.append(ct_v); ans_runs.append(an_v)
        lens.extend(len(r.answer) for r in records if r.answer)
        n = len(records)
        per_run.append({
            "run": path.name[:-len("_outputs.jsonl")],
            "n": n,
            "em": sum(em_v) / n if n else 0.0,
            "f1": sum(f1_v) / n if n else 0.0,
            "containment": sum(ct_v) / n if n else 0.0,
            "answered": sum(an_v) / n if n else 0.0,
        })

    n_pooled = sum(len(v) for v in em_runs)

    def rate(vecs: list[list[float]], counted: bool) -> dict:
        k = sum(sum(v) for v in vecs)
        out = {
            "pooled": k / n_pooled if n_pooled else 0.0,
            "n": n_pooled,
        }
        if counted:
            out["wilson95"] = wilson_interval(int(round(k)), n_pooled)
        if len(vecs) > 1:
            out["cluster_boot95"] = cluster_bootstrap_ci(vecs)
        return out

    return {
        "arm": name,
        "runs": len(runs),
        "n_pooled": n_pooled,
        "em": rate(em_runs, counted=True),
        "f1": rate(f1_runs, counted=False),
        "containment": rate(cont_runs, counted=True),
        "answered": rate(ans_runs, counted=True),
        "mean_answer_len": sum(lens) / len(lens) if lens else 0.0,
        "per_run": per_run,
        "_records": [r for run in runs for r in run],
    }


def parity(a: list[Record], b: list[Record]) -> dict:
    """Token-Jaccard agreement on shared questions (agreement, not correctness)."""
    by_q_a = {r.question: r.answer for r in a if r.question and r.answer}
    by_q_b = {r.question: r.answer for r in b if r.question and r.answer}
    shared = sorted(set(by_q_a) & set(by_q_b))
    ratios = []
    matched = 0
    for q in shared:
        ta = set(normalize_answer(by_q_a[q]).split())
        tb = set(normalize_answer(by_q_b[q]).split())
        if not ta or not tb:
            continue
        r = len(ta & tb) / len(ta | tb)
        ratios.append(r)
        matched += r >= 0.5
    return {
        "shared": len(shared),
        "matched_ge_050": matched,
        "avg_jaccard": sum(ratios) / len(ratios) if ratios else 0.0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_rate(m: dict) -> str:
    s = f"{m['pooled']:.3f}"
    if "wilson95" in m:
        lo, hi = m["wilson95"]
        s += f" (Wilson95 [{lo:.3f}, {hi:.3f}])"
    if "cluster_boot95" in m:
        lo, hi = m["cluster_boot95"]
        s += f" (run-boot95 [{lo:.3f}, {hi:.3f}])"
    return s


def render_report(scored: list[dict], parities: list[dict]) -> str:
    lines = ["# Answer-quality report (EM / token-F1)", ""]
    lines.append("Generated by `evaluation/scripts/score_quality.py`. "
                 "EM and token-F1 use SQuAD-style normalization and are the accuracy of record; "
                 "containment is a verbosity-gameable secondary signal, shown with mean answer length. "
                 "Unanswered queries score 0 on every metric (rates are over all queries).")
    lines.append("")
    for s in scored:
        lines.append(f"## Arm: {s['arm']}  (runs={s['runs']}, pooled N={s['n_pooled']})")
        lines.append("")
        lines.append(f"- **EM**: {_fmt_rate(s['em'])}")
        lines.append(f"- **F1**: {_fmt_rate(s['f1'])}")
        lines.append(f"- answered: {_fmt_rate(s['answered'])}")
        lines.append(f"- containment (secondary): {_fmt_rate(s['containment'])} "
                     f"| mean answer length: {s['mean_answer_len']:.0f} chars")
        if s["runs"] > 1:
            lines.append("- raw per-run values (EM / F1 / containment / answered):")
            for r in s["per_run"]:
                lines.append(f"  - `{r['run']}` (n={r['n']}): "
                             f"{r['em']:.3f} / {r['f1']:.3f} / {r['containment']:.3f} / {r['answered']:.3f}")
        lines.append("")
    if parities:
        lines.append("## Cross-arm parity (agreement, not correctness)")
        lines.append("")
        for p in parities:
            lines.append(f"- **{p['a']} vs {p['b']}**: {p['shared']} shared questions, "
                         f"{p['matched_ge_050']} matched (Jaccard ≥ 0.5), "
                         f"avg Jaccard = {p['avg_jaccard']:.2f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    global RESULTS_DIR, REPORT_MD, REPORT_JSON
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("arms", nargs="+",
                    help="ARM=GLOB (glob over *_outputs.jsonl basenames) or bare label")
    ap.add_argument("--results-dir", default=None,
                    help=f"override results dir (default {RESULTS_DIR})")
    args = ap.parse_args()

    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        REPORT_MD = RESULTS_DIR / "quality_report.md"
        REPORT_JSON = RESULTS_DIR / "quality_report.json"

    arms = resolve_arms(args.arms)
    scored = [score_arm(name, paths) for name, paths in arms.items()]

    parities = []
    names = list(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = parity(scored[i]["_records"], scored[j]["_records"])
            p.update(a=names[i], b=names[j])
            parities.append(p)

    report = render_report(scored, parities)
    REPORT_MD.write_text(report, encoding="utf-8")
    for s in scored:
        s.pop("_records")
    REPORT_JSON.write_text(json.dumps(
        {"arms": scored, "parity": parities}, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_MD} and {REPORT_JSON}\n")

    for s in scored:
        print(f"  {s['arm']:24s} N={s['n_pooled']:4d}  "
              f"EM={s['em']['pooled']:.3f}  F1={s['f1']['pooled']:.3f}  "
              f"answered={s['answered']['pooled']:.3f}  "
              f"containment={s['containment']['pooled']:.3f}  "
              f"len={s['mean_answer_len']:.0f}")


if __name__ == "__main__":
    main()

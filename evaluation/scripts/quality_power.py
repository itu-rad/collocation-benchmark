"""Paired quality power analysis — does N=120 questions suffice, or do we need N=200?

The RAG case studies make an **equivalence** claim: decomposing / replicating /
sharing the pipeline does not change answer quality. An equivalence claim is only
as strong as the tightness of the confidence interval on the *paired* per-question
quality difference between two arms. This script quantifies that tightness and
decides the sample size:

  * pairs the two arms on the (normalized) question text — the SelfRAG `query_id`
    is regenerated per run and is NOT stable across arms, so it cannot be the key;
  * scores every shared question with the exact EM / token-F1 metrics of record
    (imported from `score_quality`, SQuAD normalization);
  * EM: McNemar mid-p exact test on the discordant pairs (b, c) — the only pairs
    that carry information about a quality difference;
  * ΔEM and ΔF1: paired (question-level) bootstrap 95% CI, so the interval
    respects the pairing instead of treating the arms as independent samples;
  * TOST-style equivalence read at a margin δ: is the whole CI inside [-δ, +δ]?
  * sample-size projection from the observed paired-difference SD:
        half-width h(N) ≈ z · s_d / sqrt(N),  N_needed = (z · s_d / δ)^2
    reported at the current N, at N=200, and the N required to certify equivalence.

Usage:
    python evaluation/scripts/quality_power.py \
        --results-dir evaluation/collect/results/mlx_prev_1784562733 \
        --pair 'factoid: monolith vs decomposed' \
        --pair 'factoid: monolith vs decomposed_shared' \
        --pair 'multihop: monolith vs decomposed'

    # or, with no --pair, auto-discovers every monolith-vs-<other> contrast
    # per task under the results dir.

Reads `<results-dir>/e4_<task>_<arm>_quality_<dev>_r1_outputs.jsonl`. Writes a
markdown report next to the results and prints a one-line verdict per contrast.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from score_quality import normalize_answer, exact_match, token_f1, _best_answer  # noqa: E402

import json  # noqa: E402

N_BOOT = 10_000
BOOT_SEED = 1234
Z95 = 1.959963984540054

# Equivalence margins (absolute, on the 0..1 metric scale). A CI that fits
# inside [-margin, +margin] certifies the arms as quality-equivalent.
EM_MARGIN = 0.05
F1_MARGIN = 0.05

_WS = re.compile(r"\s+")


def _qnorm(text: str | None) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def load_scored(path: Path) -> dict[str, tuple[float, float]]:
    """question(normalized) -> (EM, F1). Unanswered scores 0 on both."""
    out: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q = _qnorm(r.get("question"))
        if not q:
            continue
        goldens = [str(g) for g in (r.get("golden_answers") or [])]
        ans = _best_answer(r)
        em = float(exact_match(ans, goldens)) if ans else 0.0
        f1 = token_f1(ans, goldens) if ans else 0.0
        out[q] = (em, f1)
    return out


def mcnemar_midp(b: int, c: int) -> float:
    """Two-sided McNemar mid-p exact test on discordant counts b, c.

    mid-p halves the probability of the observed table — less conservative than
    the exact binomial, the standard recommendation for paired binary data.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # exact two-sided tail (binomial, p=0.5) then subtract half the point mass
    cum = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    point = math.comb(n, k) / (2 ** n)
    p = 2 * (cum - 0.5 * point)
    return min(1.0, max(0.0, p))


def paired_bootstrap_ci(diffs: list[float], n_boot: int = N_BOOT,
                        seed: int = BOOT_SEED) -> tuple[float, float]:
    """Percentile 95% CI for the mean of paired per-question differences."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        stats.append(s / n)
    stats.sort()
    return (stats[int(0.025 * n_boot)], stats[min(int(0.975 * n_boot), n_boot - 1)])


def analyze_pair(task: str, arm_a: str, arm_b: str, results_dir: Path,
                 dev: str) -> dict | None:
    def find(arm: str) -> Path | None:
        g = glob.glob(str(results_dir / f"e4_{task}_{arm}_quality_{dev}_r1_outputs.jsonl"))
        return Path(g[0]) if g else None

    pa, pb = find(arm_a), find(arm_b)
    if not pa or not pb:
        return {"task": task, "a": arm_a, "b": arm_b, "error":
                f"missing outputs ({'a' if not pa else ''}{'b' if not pb else ''})"}

    sa, sb = load_scored(pa), load_scored(pb)
    shared = sorted(set(sa) & set(sb))
    if not shared:
        return {"task": task, "a": arm_a, "b": arm_b, "error": "no shared questions"}

    n = len(shared)
    # EM discordance (b: a-right/b-wrong, c: a-wrong/b-right). ΔEM = (c-b)/n.
    b = sum(1 for q in shared if sa[q][0] == 1 and sb[q][0] == 0)
    c = sum(1 for q in shared if sa[q][0] == 0 and sb[q][0] == 1)
    em_diffs = [sb[q][0] - sa[q][0] for q in shared]
    f1_diffs = [sb[q][1] - sa[q][1] for q in shared]

    d_em = sum(em_diffs) / n
    d_f1 = sum(f1_diffs) / n
    em_ci = paired_bootstrap_ci(em_diffs)
    f1_ci = paired_bootstrap_ci(f1_diffs)
    p_mcnemar = mcnemar_midp(b, c)

    def sd(xs: list[float], mean: float) -> float:
        if len(xs) < 2:
            return 0.0
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))

    s_em = sd(em_diffs, d_em)
    s_f1 = sd(f1_diffs, d_f1)

    def project(s_d: float, margin: float) -> dict:
        # half-width h(N) = z * s_d / sqrt(N); N to reach h = margin
        def hw(N: int) -> float:
            return Z95 * s_d / math.sqrt(N) if N > 0 else float("inf")
        n_needed = math.ceil((Z95 * s_d / margin) ** 2) if margin > 0 else 0
        return {"sd": s_d, "hw_current": hw(n), "hw_200": hw(200),
                "n_needed": n_needed, "margin": margin}

    proj_em = project(s_em, EM_MARGIN)
    proj_f1 = project(s_f1, F1_MARGIN)

    def equiv(ci: tuple[float, float], margin: float) -> bool:
        return ci[0] >= -margin and ci[1] <= margin

    return {
        "task": task, "a": arm_a, "b": arm_b, "n": n,
        "em_a": sum(sa[q][0] for q in shared) / n,
        "em_b": sum(sb[q][0] for q in shared) / n,
        "f1_a": sum(sa[q][1] for q in shared) / n,
        "f1_b": sum(sb[q][1] for q in shared) / n,
        "d_em": d_em, "em_ci": em_ci, "mc_b": b, "mc_c": c, "p_mcnemar": p_mcnemar,
        "d_f1": d_f1, "f1_ci": f1_ci,
        "proj_em": proj_em, "proj_f1": proj_f1,
        "em_equiv": equiv(em_ci, EM_MARGIN), "f1_equiv": equiv(f1_ci, F1_MARGIN),
    }


def discover_pairs(results_dir: Path, dev: str) -> list[tuple[str, str, str]]:
    pairs = []
    for task in ("factoid", "multihop"):
        files = glob.glob(str(results_dir / f"e4_{task}_*_quality_{dev}_r1_outputs.jsonl"))
        arms = []
        for f in files:
            m = re.search(rf"e4_{task}_(.+?)_quality_{dev}_r1_outputs\.jsonl$", os.path.basename(f))
            if m:
                arms.append(m.group(1))
        if "monolith" not in arms:
            continue
        for arm in sorted(a for a in arms if a != "monolith"):
            pairs.append((task, "monolith", arm))
    return pairs


def render(rows: list[dict], dev: str) -> str:
    out = ["# Paired quality power analysis (decision 5: N=120 vs N=200)", ""]
    out.append(f"Device `{dev}`. Arms paired on normalized question text "
               f"(query_id is per-run, not arm-stable). EM via McNemar mid-p; "
               f"ΔEM/ΔF1 via paired question-level bootstrap ({N_BOOT} resamples, "
               f"seed {BOOT_SEED}). Equivalence margins: EM ±{EM_MARGIN}, F1 ±{F1_MARGIN}. "
               f"`N_needed` = questions to shrink the paired-diff 95% half-width to "
               f"the margin.")
    out.append("")
    out.append("> **R=1 sufficiency for quality**: greedy decoding (`do_sample=false`) is "
               "empirically deterministic on this setup — two independent runs of the "
               "factoid monolith cell (2026-07-14 vs 2026-07-20, distinct query_ids) "
               "produced **byte-identical answers on all 120 questions**. So EM/F1 have "
               "~zero *between-run* variance and the only variance is *within-run* "
               "(question-level), which this bootstrap fully captures. `N_needed` is "
               "therefore the ACTUAL required sample size, not a lower bound, and R=1 is "
               "sufficient for the quality claim. (Latency/throughput metrics DO retain "
               "run variance — thermal/scheduling — so that R=1 caveat still applies to "
               "the performance results, just not to accuracy.)")
    out.append("")
    for r in rows:
        title = f"## {r['task']}: {r['a']} vs {r['b']}"
        if "error" in r:
            out += [title, "", f"- **skipped**: {r['error']}", ""]
            continue
        out.append(title)
        out.append("")
        out.append(f"- N (shared questions): **{r['n']}**")
        out.append(f"- EM: {r['a']}={r['em_a']:.3f}  {r['b']}={r['em_b']:.3f}  "
                   f"ΔEM={r['d_em']:+.3f}  95% CI [{r['em_ci'][0]:+.3f}, {r['em_ci'][1]:+.3f}]")
        out.append(f"  - McNemar discordant b={r['mc_b']} c={r['mc_c']}, mid-p={r['p_mcnemar']:.3f} "
                   f"→ {'no detectable EM difference' if r['p_mcnemar'] >= 0.05 else 'EM DIFFERENCE detected'}")
        out.append(f"  - equivalence @±{EM_MARGIN}: "
                   f"**{'CERTIFIED' if r['em_equiv'] else 'NOT certified'}** at N={r['n']}")
        pe = r["proj_em"]
        out.append(f"  - paired-diff SD={pe['sd']:.3f}; half-width now={pe['hw_current']:.3f}, "
                   f"@N=200={pe['hw_200']:.3f}; N_needed for ±{EM_MARGIN} = **{pe['n_needed']}**")
        out.append(f"- F1: {r['a']}={r['f1_a']:.3f}  {r['b']}={r['f1_b']:.3f}  "
                   f"ΔF1={r['d_f1']:+.3f}  95% CI [{r['f1_ci'][0]:+.3f}, {r['f1_ci'][1]:+.3f}]")
        out.append(f"  - equivalence @±{F1_MARGIN}: "
                   f"**{'CERTIFIED' if r['f1_equiv'] else 'NOT certified'}** at N={r['n']}")
        pf = r["proj_f1"]
        out.append(f"  - paired-diff SD={pf['sd']:.3f}; half-width now={pf['hw_current']:.3f}, "
                   f"@N=200={pf['hw_200']:.3f}; N_needed for ±{F1_MARGIN} = **{pf['n_needed']}**")
        out.append("")
    # aggregate verdict
    real = [r for r in rows if "error" not in r]
    if real:
        max_needed = max(max(r["proj_em"]["n_needed"], r["proj_f1"]["n_needed"]) for r in real)
        all_equiv_120 = all(r["em_equiv"] and r["f1_equiv"] for r in real)
        out.append("## Verdict")
        out.append("")
        if all_equiv_120:
            out.append(f"- Every contrast is already equivalence-**CERTIFIED at N=120**. "
                       f"N=200 is not required for the equivalence claim.")
        else:
            out.append(f"- Not all contrasts certify at N=120. Largest N required across "
                       f"contrasts/metrics = **{max_needed}**. "
                       f"{'N=200 suffices.' if max_needed <= 200 else 'Even N=200 is insufficient — see per-contrast N_needed.'}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dev", default="mlx")
    ap.add_argument("--pair", action="append", default=[],
                    help="'<task>: <armA> vs <armB>' (repeatable). Empty = auto-discover.")
    ap.add_argument("--out", default=None, help="markdown report path")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    if args.pair:
        pairs = []
        for spec in args.pair:
            task, _, rest = spec.partition(":")
            a, _, b = rest.partition(" vs ")
            pairs.append((task.strip(), a.strip(), b.strip()))
    else:
        pairs = discover_pairs(rd, args.dev)
        if not pairs:
            sys.exit(f"no monolith contrasts discovered under {rd} for dev={args.dev}")

    rows = [analyze_pair(t, a, b, rd, args.dev) for t, a, b in pairs]
    report = render(rows, args.dev)
    out_path = Path(args.out) if args.out else rd / f"quality_power_{args.dev}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}\n")
    for r in rows:
        if "error" in r:
            print(f"  {r['task']:8s} {r['a']} vs {r['b']}: SKIP ({r['error']})")
            continue
        verdict = "EQUIV@120" if (r["em_equiv"] and r["f1_equiv"]) else \
                  f"need N≤{max(r['proj_em']['n_needed'], r['proj_f1']['n_needed'])}"
        print(f"  {r['task']:8s} {r['a']:9s} vs {r['b']:16s} N={r['n']:3d}  "
              f"ΔEM={r['d_em']:+.3f}{r['em_ci']}  ΔF1={r['d_f1']:+.3f}  "
              f"mcnemar-p={r['p_mcnemar']:.3f}  {verdict}")


if __name__ == "__main__":
    main()

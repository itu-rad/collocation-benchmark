"""Unit tests for score_quality.py — hand-computed EM/F1/Wilson values.

Run:  python -m pytest evaluation/scripts/test_score_quality.py -q
 or:  python evaluation/scripts/test_score_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_quality import (  # noqa: E402
    cluster_bootstrap_ci,
    containment,
    exact_match,
    load_run,
    normalize_answer,
    token_f1,
    wilson_interval,
)

APPROX = 1e-9


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --- normalization ----------------------------------------------------------

def test_normalize_articles_punct_case():
    assert normalize_answer("The  Cat, sat!") == "cat sat"
    assert normalize_answer("An apple; a day.") == "apple day"
    assert normalize_answer("...") == ""


# --- exact match ------------------------------------------------------------

def test_em_article_and_punct_invariant():
    assert exact_match("The sixteenth president.", ["sixteenth president"]) == 1
    assert exact_match("Abraham Lincoln", ["abraham lincoln!"]) == 1
    assert exact_match("Lincoln was the sixteenth president", ["sixteenth president"]) == 0


def test_em_multiple_goldens_any_match():
    assert exact_match("yes", ["no", "yes"]) == 1
    assert exact_match("maybe", ["no", "yes"]) == 0


# --- token F1 ---------------------------------------------------------------

def test_f1_partial_overlap():
    # NOTE: avoid 'a'/'an'/'the' as tokens — normalization strips articles.
    # answer tokens {y,z,w}, golden {x,y,z}: common=2, P=2/3, R=2/3, F1=2/3
    assert approx(token_f1("y z w", ["x y z"]), 2 / 3)


def test_f1_exact_is_one_and_disjoint_is_zero():
    assert approx(token_f1("paris", ["Paris."]), 1.0)
    assert approx(token_f1("london", ["paris"]), 0.0)


def test_f1_max_over_goldens():
    # vs "x y z": 2/3 ; vs "y z": common=2, P=2/3, R=1, F1=0.8 -> max 0.8
    assert approx(token_f1("y z w", ["x y z", "y z"]), 0.8)


def test_f1_multiset_counts():
    # answer "x x y" vs golden "x y y": common = min counts = x:1, y:1 = 2
    # P = 2/3, R = 2/3 -> F1 = 2/3
    assert approx(token_f1("x x y", ["x y y"]), 2 / 3)


def test_f1_article_stripping_applies_to_goldens_too():
    # golden "a b c" loses the article 'a' -> "b c"; answer "b c d":
    # common=2, P=2/3, R=1 -> F1=0.8 (documents the normalization interplay)
    assert approx(token_f1("b c d", ["a b c"]), 0.8)


def test_f1_articles_only_answer_vs_nonempty_golden_is_zero():
    # "the" normalizes to empty; golden nonempty -> F1 0 (not 1)
    assert approx(token_f1("the", ["paris"]), 0.0)


# --- containment (secondary) ------------------------------------------------

def test_containment_substring_but_not_em():
    ans = "Lincoln was the sixteenth president of the united states"
    assert containment(ans, ["sixteenth president"]) == 1
    assert exact_match(ans, ["sixteenth president"]) == 0


# --- Wilson -----------------------------------------------------------------

def test_wilson_8_of_10():
    lo, hi = wilson_interval(8, 10)
    assert approx(lo, 0.4901625, 1e-4)
    assert approx(hi, 0.9433178, 1e-4)


def test_wilson_edge_cases():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert hi > 0.999 and lo > 0.65  # hi is 1.0 up to fp error


# --- cluster bootstrap ------------------------------------------------------

def test_cluster_bootstrap_degenerate_and_spread():
    # identical runs -> interval collapses to the point
    lo, hi = cluster_bootstrap_ci([[1.0, 1.0], [1.0, 1.0]], n_boot=500)
    assert approx(lo, 1.0) and approx(hi, 1.0)
    # heterogeneous runs -> interval spans between run means
    lo, hi = cluster_bootstrap_ci([[0.0] * 10, [1.0] * 10], n_boot=2000)
    assert lo <= 0.05 and hi >= 0.95


# --- end-to-end on a tiny fixture --------------------------------------------

def test_load_run_and_error_markers(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture_outputs.jsonl"
        rows = [
            {"question": "q1", "golden_answers": ["paris"],
             "generated_answer": "Paris."},
            {"question": "q2", "golden_answers": ["yes"],
             "generated_answer": None,
             "final_data": "Error: No more retries left"},
            {"question": "q3", "golden_answers": ["four"],
             "generated_answer": "", "final_data": "four"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        recs = load_run(p)
        assert len(recs) == 3
        assert recs[0].answer == "Paris."
        assert recs[1].answer is None          # error marker -> unanswered
        assert recs[2].answer == "four"        # falls back to final_data
        assert exact_match(recs[0].answer, recs[0].goldens) == 1
        assert exact_match(recs[2].answer, recs[2].goldens) == 1


def _run_all():
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

"""Unit tests for the pilot package (pilot_lib + apply_knobs surgery).

Run:  python evaluation/pilots/test_pilots.py
 or:  python -m pytest evaluation/pilots/test_pilots.py -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from apply_knobs import find_loadgen_blocks, render_block, strip_loadgen  # noqa: E402
from pilot_lib import detect_warmup, parse_arrivals, service_stats  # noqa: E402


# --- warm-up detection --------------------------------------------------------

def test_warmup_flat_series_k_zero():
    r = detect_warmup([1.0] * 40, window=5, epsilon=0.10)
    assert r.converged and r.k_star == 0 and r.k_fixed == 1


def test_warmup_ramp_then_flat():
    # 6 warm queries at 3x steady state, then flat
    x = [3.0] * 6 + [1.0] * 34
    r = detect_warmup(x, window=5, epsilon=0.10)
    assert r.converged
    assert 4 <= r.k_star <= 8          # onset near the transition
    assert r.k_fixed == 2 * r.k_star


def test_warmup_first_call_outlier_reported_not_folded():
    # single 10x first-call spike (ANE-compile class), otherwise flat
    x = [10.0] + [1.0] * 39
    r = detect_warmup(x, window=5, epsilon=0.10)
    assert r.converged
    assert r.outlier_idxs == [0]
    assert r.k_fixed <= 2              # spike must not inflate k


def test_warmup_too_short_and_unflat_tail_inconclusive():
    assert not detect_warmup([1.0] * 5).converged
    drift = [1.0 + 0.05 * i for i in range(40)]   # monotone drift, never flat
    assert not detect_warmup(drift, window=5, epsilon=0.10).converged


# --- service stats ------------------------------------------------------------

def test_service_stats_drops_warmup_and_outliers():
    x = [9.0, 9.0] + [1.0] * 20
    s = service_stats(x, warmup_k=2)
    assert s["n"] == 20 and abs(s["median"] - 1.0) < 1e-9
    s2 = service_stats([5.0] + [1.0] * 21, warmup_k=0, outlier_idxs=[0])
    assert s2["n"] == 21 and abs(s2["max"] - 1.0) < 1e-9


# --- arrivals sidecar ---------------------------------------------------------

def test_parse_arrivals_and_rates():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x_arrivals.csv"
        rows = ["epoch,intended_ts,actual_ts,block_s"]
        for i in range(11):
            rows.append(f"{i},{100 + i * 0.5:.6f},{100 + i * 0.5:.6f},0.000100")
        rows[5] = "4,102.000000,102.300000,0.300000"   # one blocked put
        p.write_text("\n".join(rows))
        a = parse_arrivals(p)
        assert a.n == 11
        assert a.blocked_puts == 1
        assert abs(a.intended_rate() - 2.0) < 1e-6     # 1 query / 0.5 s


# --- loadgen block surgery -----------------------------------------------------

FIXTURE = """\
name: "fixture"
pipelines:
  - name: P1
    inputs: [0]
    outputs: [1]
    dataset_stage_id: 0
    loadgen:
      component: loadgen.PoissonLoadScheduler
      queue_depth: 50
      max_queries: 10
      timeout: 600000
      config:
        # stale hand comment that must be replaced
        rate: 1.0
    stages:
      - name: A
        id: 0
        outputs: [1]
        component: stages.Stage
      - name: B
        id: 1
        component: stages.TerminalCapture
"""


def test_find_and_replace_loadgen_block_preserves_everything_else():
    lines = FIXTURE.splitlines(keepends=True)
    blocks = find_loadgen_blocks(lines)
    assert len(blocks) == 1
    start, end, indent = blocks[0]
    assert lines[start].strip() == "loadgen:"
    assert "rate: 1.0" in "".join(lines[start:end])

    before = yaml.safe_load(FIXTURE)
    fields = {
        "config.rate": (0.033, "R-LAMBDA-BELOW-SAT",
                        {"median_service_s": 18.2, "rho": 0.6, "pilot": "x"}),
        "max_queries": (40, "R-NTIMING", {}),
        "queue_depth": (40, "R-QDEPTH", {}),
        "timeout": (6500, "R-TIMEOUT", {}),
    }
    new_lines = list(lines)
    new_lines[start:end] = render_block(before["pipelines"][0]["loadgen"],
                                        fields, indent, "abc1234")
    after = yaml.safe_load("".join(new_lines))
    # non-loadgen content byte-identical semantics
    assert strip_loadgen(before) == strip_loadgen(after)
    lg = after["pipelines"][0]["loadgen"]
    assert lg["component"] == "loadgen.PoissonLoadScheduler"  # preserved
    assert lg["max_queries"] == 40 and lg["queue_depth"] == 40
    assert lg["timeout"] == 6500 and lg["config"]["rate"] == 0.033
    text = "".join(new_lines)
    assert "[knobs]" in text and "stale hand comment" not in text


def test_real_configs_block_detection():
    """Every committed experiment config's loadgen block(s) must be findable."""
    repo = HERE.parent.parent
    targets = (list((repo / "evaluation/self_rag/configs").glob("*.yml"))
               + [repo / "pipeline_configs/multimodal_vqa_mapping_a.yml",
                  repo / "pipeline_configs/torchvision_mixed.yml",
                  repo / "pipeline_configs/rag_serve_plain.yml"])
    for t in targets:
        text = t.read_text(encoding="utf-8")
        n_pipes = len(yaml.safe_load(text)["pipelines"])
        blocks = find_loadgen_blocks(text.splitlines(keepends=True))
        assert len(blocks) == n_pipes, f"{t.name}: {len(blocks)} != {n_pipes}"


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

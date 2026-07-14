"""Logic-level tests for the first-token (TTFT) trace instrumentation.

No model is loaded and no GPU/ANE is touched: the HF hook is exercised with a
mock generate loop, the MLX path with a fake stream_generate generator, and
the emitted rows are round-tripped through utils/logger.py PERF_FORMAT and —
when evaluation/contention/staged_lib.py is present (it is developed
untracked alongside analyze_staged.py) — through the real trace parser.

Run with the benchmark env:
    conda run -n benchmark_macos python -m unittest tests.test_first_token_instrumentation -v
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.logger import PERF_FORMAT, install_perf_clock  # noqa: E402
from stages.stage import (  # noqa: E402
    log_phase_single,
    log_first_token,
    log_generated_tokens,
)


def _import_staged_lib():
    """The staged-contention parser, if available (untracked module)."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation", "contention"))
    try:
        import staged_lib
        return staged_lib
    except ImportError:
        return None


class _CaptureRows:
    """Context manager: capture 'benchmark' records formatted as trace rows."""

    def __enter__(self):
        install_perf_clock()
        self.records = []
        self._handler = logging.Handler()
        self._handler.emit = self.records.append
        self._logger = logging.getLogger("benchmark")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        return False

    def rows(self):
        fmt = logging.Formatter(PERF_FORMAT)
        return [fmt.format(r) for r in self.records]


def _fake_stage(disable_logs=False):
    return SimpleNamespace(disable_logs=disable_logs,
                           parent_name="RAG pipeline",
                           name="LLM generation")


class TestRowFormat(unittest.TestCase):
    """The emitted rows must match staged_lib's documented stage-row layout:
    wall, <pipeline>, <stage>, <phase>, start|end, perf  (6 fields)."""

    def test_first_token_row_layout(self):
        stage = _fake_stage()
        with _CaptureRows() as cap:
            log_first_token(stage, "start")
            log_first_token(stage, "end")
            log_generated_tokens(stage, 57)
        rows = cap.rows()
        self.assertEqual(len(rows), 3)
        for row, phase, ev in zip(rows, ("first_token", "first_token",
                                         "n_generated_tokens"),
                                  ("start", "end", "57")):
            parts = [p.strip() for p in row.split(",")]
            self.assertEqual(len(parts), 6, row)
            float(parts[0])                       # wall
            self.assertEqual(parts[1], "RAG pipeline")
            self.assertEqual(parts[2], "LLM generation")
            self.assertEqual(parts[3], phase)
            self.assertEqual(parts[4], ev)
            int(parts[5])                         # perf_counter_ns
        # perf monotonic across the three rows
        perfs = [int(r.split(",")[-1]) for r in rows]
        self.assertEqual(perfs, sorted(perfs))

    def test_disable_logs_gates_everything(self):
        stage = _fake_stage(disable_logs=True)
        with _CaptureRows() as cap:
            log_first_token(stage, "start")
            log_first_token(stage, "end")
            log_generated_tokens(stage, 57)
        self.assertEqual(cap.rows(), [])

    def test_staged_lib_roundtrip(self):
        """Full cross-check against the real Step D parser (if present)."""
        sl = _import_staged_lib()
        if sl is None:
            self.skipTest("staged_lib.py not present in this checkout")
        stage = _fake_stage()
        with _CaptureRows() as cap:
            # one query through the generator stage, as _process_query +
            # the instrumented run() would emit it
            log_phase_single(stage.parent_name, stage.name, "run", "start")
            log_first_token(stage, "start")
            log_first_token(stage, "end")
            log_generated_tokens(stage, 57)
            log_phase_single(stage.parent_name, stage.name, "run", "end")
        with tempfile.NamedTemporaryFile("w", suffix=".csv",
                                         delete=False) as f:
            f.write("\n".join(cap.rows()) + "\n")
            path = f.name
        try:
            traces = sl.parse_trace_files([path])
        finally:
            os.unlink(path)
        pt = traces["RAG pipeline"]
        self.assertEqual(len(pt.stage_execs["LLM generation"]), 1)
        key = "LLM generation::first_token"
        self.assertIn(key, pt.stage_execs)
        self.assertEqual(len(pt.stage_execs[key]), 1)
        self.assertEqual(pt.stage_unpaired.get(key, 0), 0)
        # ttft = first_token pair end - stage run start (analyze_staged
        # _ttft_from_subphases), and it must be positive and < run duration
        (run_s, run_e, _, _) = pt.stage_execs["LLM generation"][0]
        (_, ft_e, _, _) = pt.stage_execs[key][0]
        self.assertGreater(ft_e, run_s)
        self.assertGreater(run_e, ft_e)
        # the n_generated_tokens row must NOT leak into any stage key
        self.assertNotIn("LLM generation::n_generated_tokens", pt.stage_execs)


class TestHFFirstTokenCriteria(unittest.TestCase):
    """Mock generate loop over the StoppingCriteria hook (no model)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            from stages.llm_huggingface.inference import _FirstTokenCriteria
        except ImportError as e:  # torch/transformers/outlines missing
            raise unittest.SkipTest(f"HF deps unavailable: {e}")
        cls.criteria_cls = _FirstTokenCriteria
        cls.torch = __import__("torch")

    def test_fires_exactly_once_and_never_stops(self):
        torch = self.torch
        calls = []
        crit = self.criteria_cls(lambda: calls.append(time.perf_counter_ns()))
        input_ids = torch.ones((2, 8), dtype=torch.long)
        scores = torch.zeros((2, 32))
        outs = [crit(input_ids, scores) for _ in range(50)]
        self.assertEqual(len(calls), 1)
        for o in outs:
            self.assertEqual(o.dtype, torch.bool)
            self.assertEqual(tuple(o.shape), (2,))
            self.assertFalse(bool(o.any()))
        # cached tensor: no per-token allocation after the first call
        self.assertTrue(all(o is outs[1] for o in outs[1:]))

    def test_composes_with_stopping_criteria_list(self):
        torch = self.torch
        from transformers import StoppingCriteriaList
        crit = self.criteria_cls(lambda: None)
        input_ids = torch.ones((1, 4), dtype=torch.long)
        done = StoppingCriteriaList([crit])(input_ids, torch.zeros((1, 8)))
        self.assertFalse(bool(done.any()))

    def test_steady_state_overhead_negligible(self):
        torch = self.torch
        crit = self.criteria_cls(lambda: None)
        input_ids = torch.ones((1, 8), dtype=torch.long)
        scores = torch.zeros((1, 32))
        crit(input_ids, scores)  # arm
        n = 10000
        t0 = time.perf_counter_ns()
        for _ in range(n):
            crit(input_ids, scores)
        per_call_us = (time.perf_counter_ns() - t0) / n / 1000
        # one attribute check + returning a cached tensor; anything over
        # 50us/token would be a red flag against multi-ms decode steps
        self.assertLess(per_call_us, 50, f"{per_call_us:.2f} us/call")

    def test_first_invocation_emits_one_end_row(self):
        """Wire the criteria to log_first_token exactly as run() does."""
        torch = self.torch
        stage = _fake_stage()
        with _CaptureRows() as cap:
            log_first_token(stage, "start")
            crit = self.criteria_cls(lambda: log_first_token(stage, "end"))
            input_ids = torch.ones((1, 8), dtype=torch.long)
            scores = torch.zeros((1, 32))
            for _ in range(20):  # 20-token mock decode loop
                crit(input_ids, scores)
        rows = cap.rows()
        self.assertEqual(len(rows), 2)
        self.assertIn(", first_token, start,", rows[0])
        self.assertIn(", first_token, end,", rows[1])


class _FakeResponse(SimpleNamespace):
    pass


def _fake_stream(texts):
    """Mimics mlx_lm.stream_generate's yield contract (mlx_lm 0.31.3):
    one response per detokenized segment; generation_tokens is cumulative."""
    for i, t in enumerate(texts):
        yield _FakeResponse(text=t, token=i, generation_tokens=i + 1,
                            finish_reason=None if i + 1 < len(texts)
                            else "length")


class TestMLXStreamConsumption(unittest.TestCase):
    """run() must assemble exactly what mlx_lm.generate(verbose=False)
    returns: ''.join(response.text for response in stream_generate(...))."""

    @classmethod
    def setUpClass(cls):
        try:
            import stages.llm_mlx.inference as mlx_inf
        except ImportError as e:  # mlx not installed (e.g. cuda box)
            raise unittest.SkipTest(f"MLX deps unavailable: {e}")
        cls.mlx_inf = mlx_inf

    def _make_stage(self, disable_logs=False):
        inf = object.__new__(self.mlx_inf.Inference)
        inf.disable_logs = disable_logs
        inf.parent_name = "RAG pipeline"
        inf.name = "LLM generation"
        inf._mutex = None
        inf._data_model = None
        inf._model = object()
        inf._tokenizer = object()
        inf._gen_kwargs = {"max_tokens": 8}
        inf.output_queues = {3: None}
        return inf

    def test_text_identical_to_generate_and_events_emitted(self):
        from utils.schemas import Query
        inf = self._make_stage()
        segs = ["Hel", "lo", ",", " wor", "ld"]
        seen_kwargs = {}

        def fake_stream_generate(model, tokenizer, prompt=None, **kwargs):
            seen_kwargs.update(kwargs)
            return _fake_stream(segs)

        q = Query(split="eval", batch=1, query_submitted_timestamp=0.0,
                  data="prompt text")
        with mock.patch.object(self.mlx_inf, "stream_generate",
                               fake_stream_generate), _CaptureRows() as cap:
            out = inf.run(q)
        # text identical to generate(): concatenation of response.text
        self.assertEqual(q.data, ["".join(segs)])
        self.assertEqual(list(out.keys()), [3])
        # same sampler/max_tokens args forwarded
        self.assertEqual(seen_kwargs, {"max_tokens": 8})
        rows = cap.rows()
        self.assertEqual(len(rows), 3)
        self.assertIn(", first_token, start,", rows[0])
        self.assertIn(", first_token, end,", rows[1])
        self.assertIn(", n_generated_tokens, 5,", rows[2])
        perfs = [int(r.split(",")[-1]) for r in rows]
        self.assertEqual(perfs, sorted(perfs))

    def test_multi_prompt_batch_single_pair_summed_tokens(self):
        inf = self._make_stage()
        from utils.schemas import Query
        q = Query(split="eval", batch=1, query_submitted_timestamp=0.0,
                  data=["p1", "p2"])
        streams = [["a", "b"], ["c", "d", "e"]]

        def fake_stream_generate(model, tokenizer, prompt=None, **kwargs):
            return _fake_stream(streams.pop(0))

        with mock.patch.object(self.mlx_inf, "stream_generate",
                               fake_stream_generate), _CaptureRows() as cap:
            inf.run(q)
        self.assertEqual(q.data, ["ab", "cde"])
        rows = cap.rows()
        # exactly ONE first_token pair per run() call, tokens summed (2+3)
        self.assertEqual(sum(", first_token, start," in r for r in rows), 1)
        self.assertEqual(sum(", first_token, end," in r for r in rows), 1)
        self.assertEqual(sum(", n_generated_tokens, 5," in r for r in rows), 1)

    def test_disable_logs_emits_nothing(self):
        inf = self._make_stage(disable_logs=True)
        from utils.schemas import Query
        q = Query(split="eval", batch=1, query_submitted_timestamp=0.0,
                  data="p")
        with mock.patch.object(self.mlx_inf, "stream_generate",
                               lambda *a, **k: _fake_stream(["x"])), \
                _CaptureRows() as cap:
            inf.run(q)
        self.assertEqual(q.data, ["x"])
        self.assertEqual(cap.rows(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

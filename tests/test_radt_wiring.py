"""Logic-level tests for the radt listener wiring on the collection path.

Covers the three validate_pass findings, with no model load / GPU / network:

  A1  main.radt_entrypoint enters radt.run.RADTBenchmark around run_loadgen
      (that context is the ONLY place radt 0.2.29 spawns the macmon/top
      listener processes), tags the run with choreo.label, and drains radt
      (shutdown -> run FINISHED) before os._exit. radt/mlflow are mocked.
  A2  run_collection.run_cell arms the listeners in the subprocess env:
      RADT_PRESENT + RADT_LISTENER_MACMON (mlx) / RADT_LISTENER_TOP (cuda),
      matching radt/run/benchmark.py's RADT_LISTENER_{name.upper()} gate for
      MacmonThread/TOPThread — and keeps them armed for e5 *_notrace cells.
  B   run_cell wraps the staged (stage_*) mlx subprocess wait in
      AMCBandwidthSampler writing <label>_bandwidth.csv into GLOBAL_RESULTS
      (the curation loop moves that suffix); cuda cells skip the sampler.

Run with the benchmark env:
    conda run -n benchmark_macos python -m unittest tests.test_radt_wiring -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation", "collect"))

import run_collection as rc  # noqa: E402

CONFIG = "pipeline_configs/mlperf/resnet_inference.yml"


class _FakeProc:
    """Popen stand-in: records nothing, exits 0 immediately."""

    pid = 4242

    def wait(self, timeout=None):
        return 0


class _RecordingSampler:
    """AMCBandwidthSampler stand-in recording constructor args + enter/exit."""

    instances: list["_RecordingSampler"] = []

    def __init__(self, label=None, out=None, interval=0.5, raw=False):
        self.out = out
        self.entered = False
        self.exited = False
        _RecordingSampler.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True


class RunCellEnvTest(unittest.TestCase):
    """A2 + B: the subprocess env and the bandwidth sidecar in run_cell."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmp.name)
        self.captured_envs = []
        _RecordingSampler.instances = []

        def fake_popen(cmd, cwd=None, stdout=None, stderr=None,
                       start_new_session=False, env=None):
            self.captured_envs.append(env)
            return _FakeProc()

        self._patches = [
            mock.patch.object(rc.subprocess, "Popen", side_effect=fake_popen),
            mock.patch.object(rc, "AMCBandwidthSampler", _RecordingSampler),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

    def _run(self, label, device, env=None):
        cell = rc.Cell(label, CONFIG, "test", runs=1, env=env)
        rc.run_cell(cell, device, self.results_dir, force=True)
        self.assertEqual(len(self.captured_envs), 1)
        return self.captured_envs[0]

    def test_mlx_arms_macmon_listener(self):
        env = self._run("t_radtenv_a", "mlx")
        self.assertEqual(env.get("RADT_PRESENT"), "True")
        self.assertEqual(env.get("RADT_LISTENER_MACMON"), "True")
        self.assertNotIn("RADT_LISTENER_TOP", env)

    def test_cuda_arms_top_listener(self):
        env = self._run("t_radtenv_b", "cuda")
        self.assertEqual(env.get("RADT_PRESENT"), "True")
        self.assertEqual(env.get("RADT_LISTENER_TOP"), "True")
        self.assertNotIn("RADT_LISTENER_MACMON", env)

    def test_notrace_cell_keeps_listeners_armed(self):
        # e5 *_notrace: spans off (CHOREO_DISABLE_TRACING) != listeners off.
        env = self._run("t_radtenv_c_notrace", "mlx",
                        env={"CHOREO_DISABLE_TRACING": "1"})
        self.assertEqual(env.get("CHOREO_DISABLE_TRACING"), "1")
        self.assertEqual(env.get("RADT_PRESENT"), "True")
        self.assertEqual(env.get("RADT_LISTENER_MACMON"), "True")

    def test_env_names_match_installed_radt_registry(self):
        # The env var names must be exactly what _RADTBenchmark.__enter__
        # checks: RADT_LISTENER_ + registry key upper() (MacmonThread ->
        # "Macmon", TOPThread -> "TOP"). Import the installed registry when
        # available so a radt upgrade that renames a listener fails here.
        try:
            from radt.run.listeners import listeners
        except Exception:  # pragma: no cover - env without radt
            self.skipTest("radt not importable in this env")
        keys = {k.upper() for k in listeners}
        self.assertIn("MACMON", keys)
        self.assertIn("TOP", keys)

    def test_staged_mlx_run_gets_bandwidth_sampler(self):
        self._run("stage_a_test_mlx", "mlx")
        self.assertEqual(len(_RecordingSampler.instances), 1)
        s = _RecordingSampler.instances[0]
        self.assertEqual(
            s.out, rc.GLOBAL_RESULTS / "stage_a_test_mlx_mlx_r1_bandwidth.csv")
        self.assertTrue(s.entered and s.exited)

    def test_staged_cuda_run_skips_sampler(self):
        self._run("stage_a_test_cuda", "cuda")
        self.assertEqual(_RecordingSampler.instances, [])

    def test_non_staged_mlx_run_skips_sampler(self):
        self._run("e4_factoid_monolith_pipe", "mlx")
        self.assertEqual(_RecordingSampler.instances, [])


class RadtEntrypointTest(unittest.TestCase):
    """A1: RADTBenchmark wrap, choreo.label tag, and shutdown-before-exit."""

    def _invoke(self):
        import main as main_mod

        events = []

        def rec(name, ret=None):
            def _side(*a, **k):
                events.append(name)
                return ret
            return _side

        args = SimpleNamespace(
            config_file_path=os.path.join(REPO_ROOT, CONFIG),
            pipeline_id=0, serialize_override=None,
            label="t_radt_wiring", experiment_id=0)

        mock_radt = mock.MagicMock()
        bench_ctx = mock_radt.run.RADTBenchmark.return_value
        bench_ctx.__enter__ = mock.MagicMock(side_effect=rec("bench_enter"))
        bench_ctx.__exit__ = mock.MagicMock(side_effect=rec("bench_exit", False))
        mock_radt.shutdown.side_effect = rec("shutdown")
        mock_mlflow = mock.MagicMock()
        mock_mlflow.set_tag.side_effect = rec("set_tag")

        prev_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpd, \
                mock.patch.object(main_mod, "radt", mock_radt), \
                mock.patch.object(main_mod, "mlflow", mock_mlflow), \
                mock.patch.object(main_mod, "run_loadgen",
                                  side_effect=rec("run_loadgen")) as m_loadgen, \
                mock.patch.object(main_mod, "configure_sync_export"), \
                mock.patch.object(main_mod, "kill_all_servers"), \
                mock.patch.object(main_mod, "flush_traces"), \
                mock.patch.object(main_mod, "signal"), \
                mock.patch.object(os, "_exit",
                                  side_effect=rec("os_exit")) as m_exit:
            os.chdir(tmpd)  # per-run CSV log lands in tmp, not the repo
            try:
                main_mod.radt_entrypoint(args)
            finally:
                os.chdir(prev_cwd)

        return events, mock_radt, mock_mlflow, m_loadgen, m_exit

    def test_wrap_tag_and_drain(self):
        events, mock_radt, mock_mlflow, m_loadgen, m_exit = self._invoke()

        # The benchmark context (the only listener-spawning site in radt
        # 0.2.29) is entered, run_loadgen runs inside it, and the drain +
        # hard-exit happen after it closes.
        mock_radt.run.RADTBenchmark.assert_called_once_with()
        m_loadgen.assert_called_once()
        m_exit.assert_called_once_with(0)
        self.assertEqual(
            [e for e in events if e != "set_tag"],
            ["bench_enter", "run_loadgen", "bench_exit", "shutdown", "os_exit"])

        # The run is tagged with the output label, inside the context and
        # before the workload (so even a crashed run is matchable by tag).
        mock_mlflow.set_tag.assert_called_once_with(
            "choreo.label", "t_radt_wiring")
        self.assertLess(events.index("bench_enter"), events.index("set_tag"))
        self.assertLess(events.index("set_tag"), events.index("run_loadgen"))
        self.assertEqual(os.environ.get("CHOREO_OUTPUT_LABEL"),
                         "t_radt_wiring")


if __name__ == "__main__":
    unittest.main()

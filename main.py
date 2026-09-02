import argparse
import mlflow
import signal
import sys
import os
import logging
import time

import radt
from radt.run.listeners import listeners

import numpy as np
import pandas as pd
from pydantic import BaseModel
from multiprocessing import Process, Queue
from pydantic_yaml import parse_yaml_raw_as
from typing import Literal
from utils.logger import Logger, PERF_FORMAT, install_perf_clock
from loadgen import run_loadgen
from utils.orchestrator_watchdog import OrchestratorWatchdog
from utils.schemas import BenchmarkModel
from utils.server_manager import kill_all as kill_all_servers
from utils.tracing import configure_async_export, flush_traces
from utils.trace_span import report_span_count


def parse_args():
    parser = argparse.ArgumentParser(description="Colocation benchmark runner.")
    parser.add_argument(
        "config_file_path",
        type=str,
        help="path to the pipeline configuration file.",
    )
    parser.add_argument(
        "-p",
        type=int,
        dest="pipeline_id",
        default=-1,
        # help="Maximum amount of time to train for (minutes)",
    )
    parser.add_argument(
        "-e",
        type=int,
        dest="experiment_id",
        default=0,
        help="radT experiment id",
    )
    parser.add_argument(
        "--serialize",
        dest="serialize_override",
        choices=["true", "false"],
        default=None,
        help="Override pipeline's serialize_queries flag (true/false).",
    )
    parser.add_argument(
        "--label",
        dest="label",
        type=str,
        default=None,
        help="Override the per-run output filename suffix "
             "(default: pipeline name lowercased). Useful for A/B runs of "
             "the same config under different settings.",
    )
    return parser.parse_args()


def convert_listeners(listeners: list[Literal[listeners.keys()]]) -> str:
    return "+".join(listeners)


def radt_entrypoint(args):
    # Enable ASYNC MLflow span export (radt's bounded background uploader).
    # Must happen before any mlflow.start_span call (i.e. before importing/
    # constructing the pipeline) because the flag is read at exporter init.
    # The overhead drivers still call configure_sync_export explicitly when
    # they need deterministic per-span cost.
    configure_async_export()

    # Measurement knob for the framework-overhead microbenchmark: when set, turn
    # MLflow tracing into a no-op so per-stage spans (3 per stage per query) do
    # not dominate the measured cost. This isolates the framework's CORE dispatch
    # (thread wake + queue hand-off + CSV log) from the optional profiling layer.
    # Unset in all normal runs, so it has no effect on the case studies.
    if os.environ.get("CHOREO_DISABLE_TRACING", "").lower() in ("1", "true", "yes"):
        mlflow.tracing.disable()

    # Prototype (radt-owned tracing): when enabled, every span site routes to
    # radt.trace instead of mlflow — a lightweight event onto a queue drained by a
    # radt-owned child process that owns ALL mlflow span machinery. Nothing
    # mlflow/OTel span-related then runs in this (workload) process. Start it HERE
    # on the main thread — before run_loadgen spawns pipeline threads and before
    # CUDA init (fork-safety) — mirroring how RADTBenchmark starts its metrics
    # children. NOTE: do NOT disable mlflow tracing here — under fork the child
    # would inherit the disabled tracer and drop every reconstructed span; the
    # parent simply never calls mlflow.start_span (all sites route to radt.trace),
    # so no in-process span machinery spins up regardless.
    if os.environ.get("CHOREO_PROC_TRACE", "").lower() in ("1", "true", "yes"):
        radt.trace.start(experiment_id=args.experiment_id)

    # Belt-and-braces for the prior "RadT killed the subprocess before MLflow
    # drained" issue: catch SIGTERM, flush spans, then re-raise so the
    # default handler still terminates us.
    # Only the process that INSTALLED this handler may run it. radt's listeners
    # are multiprocessing.Process children, forked after this point, and a fork
    # inherits the parent's signal handlers -- so when the parent SIGTERMs the
    # listeners at teardown, this ran inside each listener child, which then
    # tried to join a trace exporter it is not the parent of:
    #   AssertionError: can only join a child process   (radt/run/trace.py)
    # and published a spurious second "spans emitted: 0". The run's data was
    # already safe, but the teardown path is exactly the one we harden.
    _owner_pid = os.getpid()

    def _on_sigterm(signum, frame):  # pylint: disable=unused-argument
        if os.getpid() != _owner_pid:
            # A forked child (listener/logger). Just die; the parent owns the
            # flush, the span count and the mlflow run.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        # Kill any local inference servers first so a RadT timeout/kill can't
        # leak a vLLM/Ollama subprocess holding the GPU. This is the path that
        # skips stage post_run(), so it must happen here.
        kill_all_servers()
        flush_traces()
        # Publish our own emitted-span count BEFORE the exporter tears down, so a
        # reader can check it against the artifact manifest. radt drops events on
        # queue overflow and only warns; we do not change that, so completeness is
        # verified from this side instead.
        report_span_count()
        radt.trace.shutdown()  # flush + join the proc-trace exporter (no-op if unused)
        # Tear down the radt listener/logger children and close the mlflow
        # run; without this a timeout-killed run stays status=RUNNING forever
        # and the listener processes only die with the process group. No-op
        # unless RADT_PRESENT is set. shutdown() end_run()s with the default
        # FINISHED status, so re-mark the run KILLED afterwards — a validator
        # matching runs by tag must not mistake a timeout kill for a clean run.
        active = mlflow.active_run()
        radt.shutdown()
        if active is not None:
            try:
                mlflow.MlflowClient().set_terminated(active.info.run_id, "KILLED")
            except Exception:  # pylint: disable=broad-except
                pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _on_sigterm)

    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

        # Apply CLI overrides on the in-memory config.
        pipeline_cfg = benchmark_config.pipelines[args.pipeline_id]
        if args.serialize_override is not None:
            pipeline_cfg.serialize_queries = (args.serialize_override == "true")

        # Configure logging
        default_label = pipeline_cfg.name.replace(" ", "_").lower()
        pipeline_name = args.label if args.label else default_label
        # Where the per-run CSV/JSONL lands. Collection harnesses point this at
        # their own experiment's results/ dir, so runs are written where they
        # belong instead of into a shared staging directory that then has to be
        # swept. Neutral env-var name: the project has no settled title.
        log_dir = os.environ.get("BENCH_OUTPUT_DIR", "evaluation/results")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{pipeline_name}.csv")
        # Also expose the chosen label to TerminalCapture via env var so the
        # JSONL filename matches the CSV filename.
        os.environ["CHOREO_OUTPUT_LABEL"] = pipeline_name

        install_perf_clock()
        formatter = logging.Formatter(PERF_FORMAT)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger("benchmark")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Ensure the per-run log handler is flushed and closed even if
        # the pipeline raises (otherwise the last few log lines are lost).
        import atexit as _atexit

        def _cleanup_log_handler():
            try:
                file_handler.flush()
                file_handler.close()
                logger.removeHandler(file_handler)
            except Exception:  # pylint: disable=broad-except
                pass

        _atexit.register(_cleanup_log_handler)

        # Parse the .yaml and send it over as mlflow params
        def build_mlflow_config(
            config: dict, data: BaseModel | list | dict, directory: str
        ) -> None:
            if isinstance(data, BaseModel):
                for k in data.model_fields_set:
                    v = getattr(data, k)
                    build_mlflow_config(config, v, f"{directory}.{k}")
            elif isinstance(data, list):
                for i, v in enumerate(data):
                    build_mlflow_config(config, v, f"{directory}:{i}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    build_mlflow_config(config, v, f"{directory}.{k}")
            else:
                config[f"{directory}"] = data

        # Route the run to the requested radT experiment BEFORE the first
        # mlflow call auto-starts a run — without this, -e is honored only on
        # the schedule path and direct-mode runs land in the Default
        # experiment. MLFLOW_EXPERIMENT_ID (set by radt for its children)
        # takes precedence to avoid fighting the orchestrator.
        if "MLFLOW_EXPERIMENT_ID" not in os.environ and args.experiment_id:
            mlflow.set_experiment(experiment_id=str(args.experiment_id))

        # Log config
        mlflow.log_artifact(args.config_file_path, "pipeline")

        mlflow_config = {}
        build_mlflow_config(
            mlflow_config, benchmark_config.pipelines[args.pipeline_id], "pipeline"
        )
        mlflow.log_params(mlflow_config)

    # Enter the radt benchmark context ourselves. On the direct `-p N` path
    # (the collection harnesses exec `main.py <cfg> -p 0`) nothing
    # else enters it, so the macmon/top listener processes never spawn —
    # _RADTBenchmark.__enter__ (radt/run/benchmark.py) is the ONLY place they
    # are started, gated on RADT_PRESENT + RADT_LISTENER_<NAME> env vars.
    # RADTBenchmark is a no-op when RADT_PRESENT is unset (manual runs and the
    # overhead drivers stay unperturbed), and it wraps a process-wide singleton,
    # so on the radt-scheduled path (`python -m radt run`, used for the
    # orchestrated multi-pipeline cells) this reuses the instance already
    # entered by radt.run.run.start_run instead of double-spawning listeners.
    with radt.run.RADTBenchmark():
        # Tag the run with the per-run output label so post-hoc validation can
        # match mlflow runs by tag instead of wall clock. The active run exists
        # by now (mlflow.log_params above auto-starts one even without radt).
        label = os.environ.get("CHOREO_OUTPUT_LABEL", "")
        mlflow.set_tag("choreo.label", label)
        # Proc-trace: hand the active run id to the radt-owned exporter so it nests
        # the reconstructed spans under this run (mlflow.sourceRun), matching the
        # in-process arm. Emitted before run_loadgen, so it precedes every span.
        if os.environ.get("CHOREO_PROC_TRACE", "").lower() in ("1", "true", "yes"):
            _ar = mlflow.active_run()
            if _ar is not None:
                radt.trace.set_run(_ar.info.run_id)
        # Descriptive server-side run name: the cell label already encodes
        # experiment/task/arm/schedule/device/run (e.g.
        # e4_factoid_monolith_pipe_mlx_r1); append the pipeline name so
        # multi-pipeline cells stay distinguishable in the UI.
        pipe_name = benchmark_config.pipelines[args.pipeline_id].name
        mlflow.set_tag("mlflow.runName",
                       f"{label} | {pipe_name}" if label else pipe_name)
        run_loadgen(benchmark_config.pipelines[args.pipeline_id])

        # Publish our emitted-span count HERE, not at teardown: below, the CSV
        # file handler is closed and RADTBenchmark.__exit__ has already
        # end_run()'d, so a tag has no run to attach to and a log row has no
        # handler to reach. The pipeline is finished and its threads joined, so
        # no further spans are emitted after this point.
        report_span_count()

    # Force-exit after the pipeline completes. Interpreter shutdown can
    # otherwise hang for many minutes on mlflow telemetry sockets in
    # CLOSE_WAIT, joblib/loky semaphores held by ChromaDB/embedders, and
    # MLX Metal teardown. We've already captured all results to disk
    # (timing CSV + TerminalCapture JSONL) so there's nothing left to lose.
    try:
        file_handler.flush()
        file_handler.close()
        logger.removeHandler(file_handler)
    except Exception:  # pylint: disable=broad-except
        pass

    # Drain MLflow trace spans BEFORE os._exit. atexit handlers don't fire
    # on os._exit, so this is the last chance to push pending spans.
    # Run closure: the RADTBenchmark with-block above already terminated the
    # listeners, flushed their metric queues, and mlflow.end_run()'d (in that
    # order — see _RADTBenchmark.__exit__ — so no in-flight metric write lands
    # on a closed run). radt.shutdown() below is the belt-and-braces drain for
    # any path where the with-block was bypassed; it is idempotent and a no-op
    # without RADT_PRESENT. Without it, os._exit would leave the run stuck
    # status=RUNNING and orphan the listener children.
    # Reap any local inference servers before os._exit (which skips atexit), so
    # the GPU is freed even if a stage's post_run() didn't run.
    kill_all_servers()
    flush_traces()
    report_span_count()   # see the note at the other shutdown site
    radt.trace.shutdown()  # flush + join the proc-trace exporter (no-op if unused)
    radt.shutdown()
    os._exit(0)


def main(args):
    process_start_time = time.time()

    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

    print("listeners", benchmark_config.listeners)

    # Watchdog: bounded cleanup if radt.schedule_external hangs in its
    # post-loop HTTPS uploads. See utils/orchestrator_watchdog.py for
    # the full rationale. Started before schedule_external, notified
    # immediately after if it returns naturally — daemon thread, so a
    # clean main() exit kills it automatically.
    watchdog = OrchestratorWatchdog(
        experiment_id=args.experiment_id,
        parent_name=benchmark_config.name,
        pipeline_names=[p.name for p in benchmark_config.pipelines],
        process_start_time=process_start_time,
    )
    watchdog.start()

    # initialize a multiprocessing-safe logger
    logger_queue = Queue()
    logger = Logger(logger_queue, benchmark_config.name)

    # build the radt schedule
    pipeline_ids = [i for i in range(len(benchmark_config.pipelines))]
    df_schedule = pd.DataFrame(np.empty(0, dtype=radt.constants.CSV_FORMAT))
    for pipeline_id in pipeline_ids:
        df_schedule.loc[pipeline_id] = pd.Series(
            {
                "Experiment": args.experiment_id,
                "Workload": 0,
                "Name": benchmark_config.pipelines[pipeline_id].name,
                "Status": "",
                "Run": "",
                "Devices": 0,
                "Collocation": "",
                "Listeners": convert_listeners(benchmark_config.listeners).lower(),
                "File": "main.py",
                # Forward the per-run flags to the inner invocation: radt
                # re-execs main.py from this Params string, so anything not
                # carried here (label, serialize override) is silently lost
                # and the run/CSV falls back to default naming.
                "Params": f"{args.config_file_path} -p {pipeline_id}"
                          + (f" --label {args.label}" if args.label else "")
                          + (f" --serialize {args.serialize_override}"
                             if args.serialize_override else ""),
            }
        )

    # execute workload
    radt.schedule_external(
        [],
        df_schedule,
        group_name=benchmark_config.name,
    )
    watchdog.notify_schedule_returned()

    # stop the logger
    logger.stop_queue_listener()


if __name__ == "__main__":
    args = parse_args()
    if args.pipeline_id != -1:
        radt_entrypoint(args)
    else:
        main(args)

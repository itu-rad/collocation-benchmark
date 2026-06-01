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
from utils.logger import Logger
from loadgen import run_loadgen
from utils.orchestrator_watchdog import OrchestratorWatchdog
from utils.schemas import BenchmarkModel
from utils.tracing import configure_sync_export, flush_traces


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
    # Force MLflow span export to run synchronously. Must happen before any
    # mlflow.start_span call (i.e. before importing/constructing the pipeline)
    # because the async/sync flag is read at exporter init time.
    configure_sync_export()

    # Belt-and-braces for the prior "RadT killed the subprocess before MLflow
    # drained" issue: catch SIGTERM, flush spans, then re-raise so the
    # default handler still terminates us.
    def _on_sigterm(signum, frame):  # pylint: disable=unused-argument
        flush_traces()
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
        log_dir = "evaluation/results"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{pipeline_name}.csv")
        # Also expose the chosen label to TerminalCapture via env var so the
        # JSONL filename matches the CSV filename.
        os.environ["CHOREO_OUTPUT_LABEL"] = pipeline_name

        formatter = logging.Formatter("%(created)f, %(message)s")
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

        # Log config
        mlflow.log_artifact(args.config_file_path, "pipeline")

        mlflow_config = {}
        build_mlflow_config(
            mlflow_config, benchmark_config.pipelines[args.pipeline_id], "pipeline"
        )
        mlflow.log_params(mlflow_config)

    run_loadgen(benchmark_config.pipelines[args.pipeline_id])

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
    # NOTE: we do NOT call mlflow.end_run() here — RadT listeners (macmon,
    # smi, top, etc.) attach to the active run and stream metrics. Calling
    # end_run from under them closes the run, drops in-flight metric
    # writes, and can also race with mlflow.log_artifact for the yaml.
    # RadT marks the run FINISHED itself via RADTBenchmark.__exit__.
    flush_traces()
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
                "Params": f"{args.config_file_path} -p {pipeline_id}",
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

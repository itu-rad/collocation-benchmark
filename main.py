import argparse

import radt

import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
from pydantic_yaml import parse_yaml_raw_as

from utils.logger import Logger
from loadgen import run_loadgen
from utils.schemas import BenchmarkModel


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
    return parser.parse_args()


def radt_entrypoint(args):
    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)
    run_loadgen(benchmark_config.pipelines[args.pipeline_id])


def main(args):
    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

    # initialize a multiprocessing-safe logger
    logger_queue = Queue()
    logger = Logger(logger_queue, benchmark_config.name)

    # build the radt schedule
    pipeline_ids = [i for i in range(len(benchmark_config.pipelines))]
    df_schedule = pd.DataFrame(np.empty(0, dtype=radt.constants.CSV_FORMAT))
    for pipeline_id in pipeline_ids:
        df_schedule.loc[pipeline_id] = pd.Series(
            {
                "Experiment": 0,
                "Workload": 0,
                "Status": "",
                "Run": "",
                "Devices": 0,
                "Collocation": "",
                "Listeners": "smi",
                "File": "main.py",
                "Params": f"{args.config_file_path} -p {pipeline_id}",
            }
        )

    # execute workload
    radt.schedule_external(["--local"], df_schedule)

    # stop the logger
    logger.stop_queue_listener()


if __name__ == "__main__":
    args = parse_args()
    if args.pipeline_id != -1:
        radt_entrypoint(args)
    else:
        main(args)

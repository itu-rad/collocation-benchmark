import argparse
import mlflow
import sys
import os
import logging

import radt

import numpy as np
import pandas as pd
from pydantic import BaseModel
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
    parser.add_argument(
        "-e",
        type=int,
        dest="experiment_id",
        default=0,
        help="radT experiment id",
    )
    return parser.parse_args()


def radt_entrypoint(args):
    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

        # Configure logging
        pipeline_name = benchmark_config.pipelines[args.pipeline_id].name.replace(" ", "_").lower()
        log_dir = "evaluation/results"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{pipeline_name}.csv")
        
        formatter = logging.Formatter("%(created)f, %(message)s")
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        
        logger = logging.getLogger("benchmark")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

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
                "Experiment": args.experiment_id,
                "Workload": 0,
                "Name": benchmark_config.pipelines[pipeline_id].name,
                "Status": "",
                "Run": "",
                "Devices": 0,
                "Collocation": "",
                "Listeners": "smi+top+dcgmi+iostat+free+macmon",
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

    # stop the logger
    logger.stop_queue_listener()


if __name__ == "__main__":
    args = parse_args()
    if args.pipeline_id != -1:
        radt_entrypoint(args)
    else:
        main(args)

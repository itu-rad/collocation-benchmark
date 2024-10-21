import argparse
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
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

    # initialize a multiprocessing-safe logger
    logger_queue = Queue()
    logger = Logger(logger_queue, benchmark_config.name)

    # start each loadgen/pipeline as a separate process
    loadgen_processes = [
        Process(target=run_loadgen, args=[pipeline_config, logger_queue])
        for pipeline_config in benchmark_config.pipelines
    ]

    # start child processes
    for loadgen_process in loadgen_processes:
        loadgen_process.start()

    # wait for child processes to finish
    for loadgen_process in loadgen_processes:
        loadgen_process.join()

    logger.stop_queue_listener()


if __name__ == "__main__":
    main()

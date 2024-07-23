import argparse
import yaml
from multiprocessing import Process, Queue

from utils.logger import Logger
from loadgen.loadgen import runLoadGen


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
    benchmark_config = None
    with open(args.config_file_path, "r", encoding="utf-8") as file:
        benchmark_config = yaml.safe_load(file)

    benchmark_name = benchmark_config.get("name", "Unknown benchmark name")

    logger_queue = Queue()
    logger = Logger(logger_queue, benchmark_name)

    # create loadgen for each pipeline
    pipeline_configs = benchmark_config.get("pipelines", [])

    # start each loadgen/pipeline as a separate process
    loadgen_processes = [
        Process(target=runLoadGen, args=[pipeline_config, logger_queue])
        for pipeline_config in pipeline_configs
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

import argparse
import uuid
import yaml

from pipeline.pipeline import Pipeline


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
    pipelines_config = None
    with open(args.config_file_path, "r") as file:
        pipelines_config = yaml.safe_load(file)

    pipelines = []
    for pipeline in pipelines_config.get("pipelines", []):
        print("Creating pipeline.")
        pipelines.append(Pipeline(pipeline, False, None))
        print("Done creating pipeline.")

    # TODO: everything past this point is the responsibility of LoadGen
    for pipeline in pipelines:
        pipeline.prepare()

    for pipeline in pipelines:
        for _ in range(100):
            result = pipeline.run(uuid.uuid4())
            print(result)


if __name__ == "__main__":
    main()

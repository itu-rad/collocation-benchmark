import yaml

from pipeline.pipeline import Pipeline


def main():
    pipelines_config = None
    with open("pipeline_configs/efficientnetv2_imagenette.yml", "r") as file:
        pipelines_config = yaml.safe_load(file)

    pipelines = []
    for pipeline in pipelines_config.get("Pipelines", []):
        pipelines.append(Pipeline(pipeline, False, None))


if __name__ == "__main__":
    main()

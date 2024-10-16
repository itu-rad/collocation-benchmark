from pydantic import BaseModel
from .pipeline import PipelineModel


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    name: str = "Unknown benchmark name"

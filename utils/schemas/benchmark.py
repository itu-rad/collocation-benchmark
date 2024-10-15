from pydantic import BaseModel
from .pipeline import PipelineModel


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    name: str = "Unknown benchmark name"
    pipelines: list[PipelineModel]

from pydantic import BaseModel
from typing import Literal
from .pipeline import PipelineModel

from radt.run.listeners import listeners


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    listeners: list[Literal[tuple(listeners.keys())]] = ["TOP"]
    name: str = "Unknown benchmark name"

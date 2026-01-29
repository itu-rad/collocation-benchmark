from pydantic import BaseModel
from typing import Literal
from .pipeline import PipelineModel

from radt.run.listeners import listeners as radt_listeners


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    listeners: list[Literal[tuple(radt_listeners.keys())]] = ["TOP"]
    name: str = "Unknown benchmark name"

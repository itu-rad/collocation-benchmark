from pydantic import BaseModel
from typing import Literal
from .pipeline import PipelineModel

from radt.run.listeners import listeners as radt_listeners


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    listeners: list[Literal[tuple([x.lower() for x in radt_listeners.keys()])]] = ["top"]
    name: str = "Unknown benchmark name"

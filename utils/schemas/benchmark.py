from pydantic import BaseModel
from typing import Literal
from .pipeline import PipelineModel

from radt.run.listeners import listeners as radt_listeners


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    # macmon is the macOS-compatible RadT listener (Apple Silicon power /
    # thermals). NVIDIA hosts should override per-config with smi+top+...
    listeners: list[Literal[tuple([x.lower() for x in radt_listeners.keys()])]] = ["macmon"]
    name: str = "Unknown benchmark name"

import sys

from pydantic import BaseModel, Field
from typing import Literal
from .pipeline import PipelineModel

from radt.run.listeners import listeners as radt_listeners


def _default_listeners() -> list[str]:
    """OS-appropriate default RadT listeners.

    macOS (Apple Silicon) only has macmon; NVIDIA/Linux hosts use the
    smi/top/iostat/dcgmi set. A config can still override `listeners` explicitly.
    """
    if sys.platform == "darwin":
        return ["macmon"]
    return ["top", "smi", "iostat", "dcgmi"]


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    # Defaults by platform (see _default_listeners). Override per-config as needed.
    listeners: list[Literal[tuple([x.lower() for x in radt_listeners.keys()])]] = Field(
        default_factory=_default_listeners
    )
    name: str = "Unknown benchmark name"

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
    # How the pipelines of this benchmark share the GPU, when there is more than
    # one. radt launches each pipeline as its own process, so a partitioning
    # mechanism has something to separate:
    #   ""      time-sliced -- the default, processes share the GPU as usual
    #   "mps"   radt starts the CUDA MPS control daemon for the group
    #   "<n>g.<m>gb"  a MIG profile string, e.g. "1g.10gb"
    # radt reads this from the schedule row main.py builds; it is declared here
    # so a collocation cell is one YAML key rather than a separate mechanism.
    collocation: str = ""
    name: str = "Unknown benchmark name"

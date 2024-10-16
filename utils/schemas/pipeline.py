from pydantic import BaseModel
from .loadgen import LoadGenModel
from .stage import StageModel


class PipelineModel(BaseModel):
    """
    Pipeline configuration parsed from yaml file.
    """

    inputs: list[int]
    outputs: list[int]
    dataset_stage_id: int
    loadgen: LoadGenModel
    stages: list[StageModel]
    name: str = "Unknown pipeline name"

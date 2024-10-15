from pydantic import BaseModel
from typing import Any


class StageModel(BaseModel):
    """
    Stage configuration parsed from yaml file.
    """

    id: int
    name: str
    outputs: list[int] = []
    component: str
    disable_logs: bool = False
    config: dict[str, Any] = {}

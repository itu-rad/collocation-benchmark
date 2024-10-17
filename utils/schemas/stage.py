from pydantic import BaseModel
from typing import Any

from stages.queues.polling import polling_policy


class StageModel(BaseModel):
    """
    Stage configuration parsed from yaml file.
    """

    id: int
    name: str
    component: str
    outputs: list[int] = []
    polling_policy: str = "stages.queues.polling.SingleQueuePolicy"
    disable_logs: bool = False
    config: dict[str, Any] = {}

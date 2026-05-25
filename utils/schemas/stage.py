from pydantic import BaseModel
from typing import Any


class StageModel(BaseModel):
    """
    Stage configuration parsed from yaml file.
    """

    id: int
    name: str
    component: str
    outputs: list[int] = []
    polling_policy: str = "utils.queues.polling.SingleQueuePolicy"
    disable_logs: bool = False
    config: dict[str, Any] = {}
    # Bound the input queue for backpressure. None == unbounded (default).
    # WARNING: Setting this on a stage that is part of a feedback cycle
    # (e.g. retry loops) can deadlock — upstream put() blocks because the
    # downstream stage is itself waiting to put() back into the upstream.
    max_input_queue_depth: int | None = None

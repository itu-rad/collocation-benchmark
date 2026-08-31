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
    # Keep query.data alive past the pipeline's output stage. Off by default:
    # once a query leaves the terminal stage nothing reads its payload, and
    # holding it means the workload's memory is reclaimed on the collector
    # thread, inside the framework's measured exit interval, rather than on the
    # stage's own thread where a monolith would pay it. Set True only for a
    # consumer that genuinely reads the payload after the pipeline.
    keep_payload_at_exit: bool = False
    # Bound the input queue for backpressure. None == unbounded (default).
    # WARNING: Setting this on a stage that is part of a feedback cycle
    # (e.g. retry loops) can deadlock — upstream put() blocks because the
    # downstream stage is itself waiting to put() back into the upstream.
    max_input_queue_depth: int | None = None

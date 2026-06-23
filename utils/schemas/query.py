from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


@dataclass
class Query:
    """
    Structure of a query, which is passed to the pipeline and flows between the stages.
    """

    split: str
    batch: int
    query_submitted_timestamp: float
    epoch: int = 0
    # Use a factory so each Query instance gets a FRESH id. A bare
    # `= uuid.uuid4()` default is evaluated once at class-definition time, so
    # every query in a process would share one id — which also collapses the
    # routers' per-query retry budgets (they key on query_id) into a single
    # global budget. See REPLICATION_NOTES.md (Hurdle 5).
    query_id: uuid.UUID = field(default_factory=uuid.uuid4)
    data: Any = None
    context: Any = None
    # Per-query flow id carried between consecutive tracing spans so Perfetto
    # can link the slices across the scheduler / pipeline / per-stage threads.
    out_flow_id: Optional[uuid.UUID] = None

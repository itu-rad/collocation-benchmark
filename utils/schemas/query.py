from dataclasses import dataclass
from typing import Any, Optional
import uuid
from mlflow.entities import Span


@dataclass
class Query:
    """
    Structure of a query, which is passed to the pipeline and flows between the stages.
    """

    split: str
    batch: int
    query_submitted_timestamp: float
    epoch: int = 0
    query_id: int = uuid.uuid4()
    data: Any = None
    context: Any = None
    trace_span: Optional[Span] = None
    loadgen_span: Optional[Span] = None

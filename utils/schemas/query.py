from dataclasses import dataclass
from typing import Any
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
    query_id: int = uuid.uuid4()
    data: Any = None
    context: Any = None

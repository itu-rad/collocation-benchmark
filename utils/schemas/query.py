from dataclasses import dataclass
from typing import Any
import uuid


@dataclass
class Query:
    """
    Structure of a query, which is passed to the pipeline and flows between the stages.
    """

    query_id: int = uuid.uuid4()
    split: str
    batch: int
    query_submitted_timestamp: float
    data: Any = None
    epoch: int = 0

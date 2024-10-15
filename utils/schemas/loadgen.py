from pydantic import BaseModel
from typing import Any


class LoadGenModel(BaseModel):
    """
    Load generation configuration.
    """

    component: str
    queue_depth: int = 10
    max_queries: int
    timeout: int
    config: dict[str, Any] = {}

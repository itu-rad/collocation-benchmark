from pydantic import BaseModel
from typing import Any


class LoadGenModel(BaseModel):
    """
    Load generation configuration.
    """

    component: str
    max_queries: int
    timeout: int
    queue_depth: int = 10
    config: dict[str, Any] = {}

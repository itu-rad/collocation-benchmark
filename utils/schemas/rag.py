from typing import Literal
from pydantic import BaseModel


class Score(BaseModel):
    score: bool


class RouterAnswer(BaseModel):
    index_backend: Literal["sqlite", "llm"]

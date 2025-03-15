from typing import Literal
from pydantic import BaseModel


class Score(BaseModel):
    score: bool


class RouterAnswer(BaseModel):
    search_engine: Literal["vectorstore", "web-search"]

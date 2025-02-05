from typing import Any

from pydantic import BaseModel


class BaseEntity(BaseModel):
    _phringe: Any = None
    name: str = None

    def __init__(self, **data):
        super().__init__(**data)
        self._phringe = data.pop("_phringe", None)

    class Config:
        arbitrary_types_allowed = True

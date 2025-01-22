from typing import Any

from pydantic import BaseModel


class BaseEntity(BaseModel):
    def __init__(self, **data):
        name: str = None
        _device: Any = None

        super().__init__(**data)

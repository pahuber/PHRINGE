from typing import Any

from pydantic import BaseModel


class BaseEntity(BaseModel):
    _device: Any
    name: str = None

    def __init__(self, **data):
        super().__init__(**data)
        self._device = data.pop("_device", None)

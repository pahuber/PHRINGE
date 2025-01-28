from typing import Any

from pydantic import BaseModel, PrivateAttr


class BaseEntity(BaseModel):
    name: str = None
    _device: Any = PrivateAttr()

    def __init__(self, **data):
        _device = data.pop("_device", None)
        super().__init__(**data)
        self._device = _device

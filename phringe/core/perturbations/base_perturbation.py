from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from phringe.core.base_entity import BaseEntity


class BasePerturbation(ABC, BaseEntity):
    _phringe: Any = None

    @property
    @abstractmethod
    def time_series(self) -> Tensor:
        pass

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor


class BasePerturbation(ABC, BaseModel):
    rms: str
    color: str

    @field_validator('color')
    def _validate_color(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the color input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The color
        """
        if value not in ['white', 'pink', 'brown']:
            raise ValueError(f'{value} is not a valid input for {info.field_name}. Must be one of white, pink, brown.')
        return value

    @abstractmethod
    def get_time_series(
            self,
            number_of_inputs: int,
            total_integration_time: float,
            number_of_simulation_time_steps: int,
            **kwargs
    ) -> Tensor:
        pass

    def _get_color_coeff(self) -> int:
        match self.color:
            case 'white':
                coeff = 0
            case 'pink':
                coeff = 1
            case 'brown':
                coeff = 2
        return coeff

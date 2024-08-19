from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.util.noise_generator import NoiseGenerator


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
            simulation_time_step_size: float,
            number_of_simulation_time_steps: int,
            **kwargs
    ) -> Tensor:
        pass

    def _get_color(self, noise_generator: NoiseGenerator):
        match self.color:
            case 'white':
                color = noise_generator.white()
            case 'pink':
                color = noise_generator.pink()
            case 'brown':
                color = noise_generator.brown()
        return color

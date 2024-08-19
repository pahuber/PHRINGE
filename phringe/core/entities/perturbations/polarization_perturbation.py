from typing import Any

import numpy as np
import torch
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.perturbations.base_perturbation import BasePerturbation
from phringe.core.entities.perturbations.noise_generator import NoiseGenerator
from phringe.io.validators import validate_quantity_units


class PolarizationPerturbation(BasePerturbation, BaseModel):

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,)).si.value

    def get_time_series(
            self,
            number_of_inputs: int,
            simulation_time_step_size: float,
            number_of_simulation_time_steps: int
    ) -> Tensor:
        time_series = np.zeros((number_of_inputs, number_of_simulation_time_steps))

        noise_generator = NoiseGenerator()
        color = self._get_color(noise_generator)

        for k in range(number_of_inputs):
            time_series[k] = noise_generator.generate(
                dt=simulation_time_step_size,
                n=number_of_simulation_time_steps,
                colour=color
            )
            time_series[k] *= self.rms / np.sqrt(np.mean(time_series[k] ** 2))

        return torch.tensor(time_series, dtype=torch.float32)

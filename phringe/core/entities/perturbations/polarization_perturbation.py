from typing import Any

import numpy as np
import torch
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from stochastic.processes import ColoredNoise
from torch import Tensor

from phringe.core.entities.perturbations.base_perturbation import BasePerturbation
from phringe.io.validators import validate_quantity_units


class PolarizationPerturbation(BasePerturbation, BaseModel):

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,)).si.value

    def get_time_series(
            self,
            number_of_inputs: int,
            total_integration_time: float,
            number_of_simulation_time_steps: int,
            **kwargs
    ) -> Tensor:
        time_series = np.zeros((number_of_inputs, number_of_simulation_time_steps))

        color_coeff = self._get_color_coeff()
        noise = ColoredNoise(color_coeff, total_integration_time)

        for k in range(number_of_inputs):
            time_series[k] = noise.sample(number_of_simulation_time_steps - 1)
            time_series[k] *= self.rms / np.sqrt(np.mean(time_series[k] ** 2))

        return torch.tensor(time_series, dtype=torch.float32)

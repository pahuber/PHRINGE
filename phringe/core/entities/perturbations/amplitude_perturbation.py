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


class AmplitudePerturbation(BasePerturbation, BaseModel):

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.percent,)).si.value

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
            time_series[k] *= self.rms / np.sqrt(
                np.mean(time_series[k] ** 2)) * 2.6  # x (empirical) 1.4 to match the final rms value
            time_series[k] = 1 + time_series[k]
            time_series[k] = np.clip(time_series[k], None, 1)

        return torch.tensor(time_series, dtype=torch.float32)

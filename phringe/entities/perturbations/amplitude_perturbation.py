from typing import Any

import astropy.units as u
import numpy as np
import torch
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.observing_entity import observing_property
from phringe.entities.perturbations.base_perturbation import BasePerturbation
from phringe.io.validators import validate_quantity_units


class AmplitudePerturbation(BasePerturbation):

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.percent,)).si.value

    @observing_property(
        observed_attributes=(
                lambda s: s._number_of_inputs,
                lambda s: s._observation.modulation_period,
                lambda s: s._number_of_simulation_time_steps
        )
    )
    def _time_series(self) -> Tensor:
        time_series = np.zeros((self._number_of_inputs, self._number_of_simulation_time_steps))

        color_coeff = self._get_color_coeff()

        for k in range(self._number_of_inputs):
            time_series[k] = self._calculate_time_series_from_psd(
                color_coeff,
                self._observation.modulation_period,
                self._number_of_simulation_time_steps
            )

        return torch.tensor(time_series, dtype=torch.float32, device=self._device)

from typing import Any

import astropy.units as u
import torch
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.entities.perturbations.base_perturbation import BasePerturbation
from phringe.io.validators import validate_quantity_units


class AmplitudePerturbation(BasePerturbation):

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.percent,)).si.value

    @property
    def _time_series(self) -> Tensor:
        time_series = torch.zeros(
            (self._phringe._instrument.number_of_inputs, len(self._phringe.simulation_time_steps)),
            dtype=torch.float32,
            device=self._phringe._device)

        if not self._has_manually_set_time_series and self.color is not None and self.rms is not None:

            color_coeff = self._get_color_coeff()

            for k in range(self._phringe._instrument.number_of_inputs):
                time_series[k] = self._calculate_time_series_from_psd(
                    color_coeff,
                    self._phringe._observation.modulation_period,
                    len(self._phringe.simulation_time_steps)
                )

        return time_series

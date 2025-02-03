# from typing import Any
from typing import Union, Any

import astropy.units as u
import numpy as np
import torch
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.entities.perturbations.base_perturbation import BasePerturbation
from phringe.io.validators import validate_quantity_units


class PhasePerturbation(BasePerturbation):
    _wavelength_bin_centers: Any = None

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.meter,))

    # OVerwrite property of base class because an additional attribute, wavelengths, is required here
    @property
    def _time_series(self) -> Union[Tensor, None]:
        time_series = torch.zeros(
            (self._phringe._instrument.number_of_inputs, len(self._phringe._instrument.wavelength_bin_centers),
             len(self._phringe.simulation_time_steps)),
            dtype=torch.float32, device=self._phringe._device)

        if not self._has_manually_set_time_series and self.color is not None and self.rms is not None:

            color_coeff = self._get_color_coeff()

            for k in range(self._phringe._instrument.number_of_inputs):
                time_series[k] = self._calculate_time_series_from_psd(
                    color_coeff,
                    self._phringe._observation.modulation_period,
                    len(self._phringe.simulation_time_steps)
                )

            for il, l in enumerate(self._phringe._instrument.wavelength_bin_centers):
                time_series[:, il] = 2 * np.pi * time_series[:, il] / l

        return time_series

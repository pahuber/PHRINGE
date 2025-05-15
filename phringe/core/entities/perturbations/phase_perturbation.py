# from typing import Any
from typing import Union, Any

import astropy.units as u
import numpy as np
import torch
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.perturbations.base_perturbation import BasePerturbation
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
            (
                self._phringe._instrument.number_of_inputs,
                len(self._phringe._instrument.wavelength_bin_centers),
                len(self._phringe.simulation_time_steps)
            ),
            dtype=torch.float32, device=self._phringe._device)

        if not self._has_manually_set_time_series and self.color_coeff is not None and self.rms is not None:

            color_coeff = self.color_coeff
            wl_bounds = self._phringe._instrument.wavelength_bands_boundaries
            num_bands = len(wl_bounds) + 1
            wavelengths = self._phringe._instrument.wavelength_bin_centers

            for j in range(num_bands):

                time_series_per_band = torch.zeros(
                    (
                        self._phringe._instrument.number_of_inputs,
                        len(self._phringe._instrument.wavelength_bin_centers),
                        len(self._phringe.simulation_time_steps)
                    ),
                    dtype=torch.float32, device=self._phringe._device)

                for k in range(self._phringe._instrument.number_of_inputs):
                    time_series_per_band[k] = self._calculate_time_series_from_psd(
                        color_coeff,
                        self._phringe._observation.modulation_period,
                        len(self._phringe.simulation_time_steps)
                    )

                # Get the index of the closest wavelength value
                if num_bands == 1:
                    index_low = 0
                    index_up = len(wavelengths)
                elif j == 0:
                    index_low = 0
                    index_up = (torch.abs(wavelengths - wl_bounds[j])).argmin()
                elif j == num_bands - 1:
                    index_low = (torch.abs(wavelengths - wl_bounds[j - 1])).argmin()
                    index_up = len(wavelengths)
                else:
                    index_low = (torch.abs(wavelengths - wl_bounds[j - 1])).argmin()
                    index_up = (torch.abs(wavelengths - wl_bounds[j])).argmin()

                time_series[:, index_low:index_up, :] = (2 * np.pi
                                                         * time_series_per_band[:, index_low:index_up, :]
                                                         / wavelengths[None, index_low:index_up, None])

        return time_series

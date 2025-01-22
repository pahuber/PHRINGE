# from typing import Any
import warnings
from typing import Union, Any

import astropy.units as u
import numpy as np
import torch
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.entities.perturbations.base_perturbation import BasePerturbation
from phringe.io.validators import validate_quantity_units
from phringe.util.warning import MissingRequirementWarning


class PhasePerturbation(BasePerturbation):
    _wavelength_bin_center: Any = None

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.meter,)).si.value

    # OVerwrite property of base class because an additional attribute, wavelengths, is required here
    @property
    def _time_series(self) -> Union[Tensor, None]:
        if self._observation is None:
            warnings.warn(MissingRequirementWarning('Observation', '_time_series'))
            return None

        if self._number_of_simulation_time_steps is None:
            warnings.warn(MissingRequirementWarning('Number of simulation time steps', '_time_series'))
            return None

        if self._wavelength_bin_centers is None:
            warnings.warn(MissingRequirementWarning('Wavelength bin centers', '_time_series'))
            return None

        return self._get_cached_value(
            compute_func=self._calculate_time_series,
            required_attributes=(
                self._number_of_inputs,
                self._observation.modulation_period if self._observation is not None else None,
                self._wavelength_bin_centers if self._wavelength_bin_centers is not None else None,
                self._number_of_simulation_time_steps
            )
        )

    def _calculate_time_series(
            self,
            # modulation_period: float,
            # number_of_simulation_time_steps: int,
            # **kwargs
    ) -> Tensor:
        time_series = np.zeros(
            (self._number_of_inputs, len(self._wavelength_bin_centers), self._number_of_simulation_time_steps))
        color_coeff = self._get_color_coeff()

        for k in range(self._number_of_inputs):
            time_series[k] = self._calculate_time_series_from_psd(
                color_coeff,
                self._observation.modulation_period,
                self._number_of_simulation_time_steps
            )

        for il, l in enumerate(self._wavelength_bin_centers):
            time_series[:, il] = 2 * np.pi * time_series[:, il] / l

        return torch.tensor(time_series, dtype=torch.float32, device=self._device)

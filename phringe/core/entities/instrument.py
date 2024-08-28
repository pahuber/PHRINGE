from functools import cached_property
from typing import Tuple, Any

import numpy as np
import torch
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.perturbations.amplitude_perturbation import AmplitudePerturbation
from phringe.core.entities.perturbations.base_perturbation import BasePerturbation
from phringe.core.entities.perturbations.phase_perturbation import PhasePerturbation
from phringe.core.entities.perturbations.polarization_perturbation import PolarizationPerturbation
from phringe.io.validators import validate_quantity_units


class _Perturbations(BaseModel):
    amplitude_perturbation: AmplitudePerturbation
    phase_perturbation: PhasePerturbation
    polarization_perturbation: PolarizationPerturbation


class Instrument(BaseModel):
    """Class representing the instrument.

    :param amplitude_perturbation_lower_limit: The lower limit of the amplitude perturbation
    :param amplitude_perturbation_upper_limit: The upper limit of the amplitude perturbation
    :param array_configuration: The array configuration
    :param aperture_diameter: The aperture diameter
    :param beam_combination_scheme: The beam combination scheme
    :param spectral_resolving_power: The spectral resolving power
    :param wavelength_range_lower_limit: The lower limit of the wavelength range
    :param wavelength_range_upper_limit: The upper limit of the wavelength range
    :param throughput: The unperturbed instrument throughput
    :param phase_perturbation_rms: The phase perturbation rms
    :param phase_falloff_exponent: The phase falloff exponent
    :param baseline_maximum: The maximum baseline
    :param baseline_minimum: The minimum baseline
    :param polarization_perturbation_rms: The polarization perturbation rms
    :param polarization_falloff_exponent: The polarization falloff exponent
    :param field_of_view: The field of view
    :param amplitude_perturbation_time_series: The amplitude perturbation time series
    :param phase_perturbation_time_series: The phase perturbation time series
    :param polarization_perturbation_time_series: The polarization perturbation time series
    """

    # amplitude_perturbation_lower_limit: float
    # amplitude_perturbation_upper_limit: float
    array_configuration_matrix: Any
    complex_amplitude_transfer_matrix: Any
    differential_outputs: Any
    baseline_maximum: str
    baseline_minimum: str
    sep_at_max_mod_eff: Any
    aperture_diameter: str
    spectral_resolving_power: int
    wavelength_range_lower_limit: str
    wavelength_range_upper_limit: str
    throughput: float
    quantum_efficiency: float
    perturbations: _Perturbations
    field_of_view: Any = None

    @field_validator('aperture_diameter')
    def _validate_aperture_diameter(cls, value: Any, info: ValidationInfo) -> Tensor:
        """Validate the aperture diameter input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The aperture diameter in units of length
        """
        return torch.tensor(
            validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value,
            dtype=torch.float32
        )

    @field_validator('baseline_minimum')
    def _validate_baseline_minimum(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the baseline minimum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The minimum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('baseline_maximum')
    def _validate_baseline_maximum(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the baseline maximum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The maximum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('wavelength_range_lower_limit')
    def _validate_wavelength_range_lower_limit(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the wavelength range lower limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The lower wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('wavelength_range_upper_limit')
    def _validate_wavelength_range_upper_limit(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the wavelength range upper limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The upper wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @cached_property
    def _wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_wavelength_bins()

    @cached_property
    def wavelength_bin_centers(self) -> np.ndarray:
        """Return the wavelength bin centers.

        :return: An array containing the wavelength bin centers
        """
        return self._wavelength_bins[0]

    @cached_property
    def wavelength_bin_widths(self) -> np.ndarray:
        """Return the wavelength bin widths.

        :return: An array containing the wavelength bin widths
        """
        return self._wavelength_bins[1]

    @cached_property
    def wavelength_bin_edges(self) -> np.ndarray:
        """Return the wavelength bin edges.

        :return: An array containing the wavelength bin edges
        """
        return torch.concatenate((self.wavelength_bin_centers - self.wavelength_bin_widths / 2,
                                  self.wavelength_bin_centers[-1:] + self.wavelength_bin_widths[-1:] / 2))

    def _get_wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the wavelength bin centers and widths. The wavelength bin widths are calculated starting from the
        wavelength lower range. As a consequence, the uppermost wavelength bin might be smaller than anticipated.

        :return: A tuple containing the wavelength bin centers and widths
        """
        current_minimum_wavelength = self.wavelength_range_lower_limit
        wavelength_bin_centers = []
        wavelength_bin_widths = []

        while current_minimum_wavelength <= self.wavelength_range_upper_limit:
            center_wavelength = current_minimum_wavelength / (1 - 1 / (2 * self.spectral_resolving_power))
            bin_width = 2 * (center_wavelength - current_minimum_wavelength)
            if (center_wavelength + bin_width / 2 <= self.wavelength_range_upper_limit):
                wavelength_bin_centers.append(center_wavelength)
                wavelength_bin_widths.append(bin_width)
                current_minimum_wavelength = center_wavelength + bin_width / 2
            else:
                last_bin_width = self.wavelength_range_upper_limit - current_minimum_wavelength
                last_center_wavelength = self.wavelength_range_upper_limit - last_bin_width / 2
                wavelength_bin_centers.append(last_center_wavelength)
                wavelength_bin_widths.append(last_bin_width)
                break
        return torch.asarray(wavelength_bin_centers, dtype=torch.float32), torch.asarray(wavelength_bin_widths,
                                                                                         dtype=torch.float32)

    def get_all_perturbations(self) -> list[BasePerturbation]:
        """Return all perturbations.

        :return: A list containing all perturbations
        """
        return [
            self.perturbations.amplitude_perturbation,
            self.perturbations.phase_perturbation,
            self.perturbations.polarization_perturbation
        ]

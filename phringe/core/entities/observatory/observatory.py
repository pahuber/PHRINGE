from functools import cached_property
from typing import Tuple, Any

import numpy as np
import torch
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.observatory.array import (
    Array,
    ArrayEnum,
    EmmaXCircularRotation,
    EmmaXDoubleStretch,
    EquilateralTriangleCircularRotation,
    RegularPentagonCircularRotation,
)
from phringe.core.entities.observatory.beam_combiner import (
    BeamCombiner,
    BeamCombinerEnum,
    DoubleBracewell,
    Kernel3,
    Kernel4,
    Kernel5,
)
from phringe.io.validators import validate_quantity_units


class Observatory(BaseModel):
    """Class representing the observatory.

    :param amplitude_perturbation_lower_limit: The lower limit of the amplitude perturbation
    :param amplitude_perturbation_upper_limit: The upper limit of the amplitude perturbation
    :param array_configuration: The array configuration
    :param aperture_diameter: The aperture diameter
    :param beam_combination_scheme: The beam combination scheme
    :param spectral_resolving_power: The spectral resolving power
    :param wavelength_range_lower_limit: The lower limit of the wavelength range
    :param wavelength_range_upper_limit: The upper limit of the wavelength range
    :param unperturbed_instrument_throughput: The unperturbed instrument throughput
    :param phase_perturbation_rms: The phase perturbation rms
    :param phase_falloff_exponent: The phase falloff exponent
    :param polarization_perturbation_rms: The polarization perturbation rms
    :param polarization_falloff_exponent: The polarization falloff exponent
    :param field_of_view: The field of view
    :param amplitude_perturbation_time_series: The amplitude perturbation time series
    :param phase_perturbation_time_series: The phase perturbation time series
    :param polarization_perturbation_time_series: The polarization perturbation time series
    """

    amplitude_perturbation_lower_limit: float
    amplitude_perturbation_upper_limit: float
    array: str
    beam_combiner: str
    aperture_diameter: str
    spectral_resolving_power: int
    wavelength_range_lower_limit: str
    wavelength_range_upper_limit: str
    unperturbed_instrument_throughput: float
    phase_perturbation_rms: str
    phase_falloff_exponent: float
    polarization_perturbation_rms: str
    polarization_falloff_exponent: float
    field_of_view: Any = None
    amplitude_perturbation_time_series: Any = None
    phase_perturbation_time_series: Any = None
    polarization_perturbation_time_series: Any = None

    def __init__(self, **data):
        """Constructor method.
        """
        super().__init__(**data)
        self.array = self._load_array(self.array)
        self.beam_combiner = self._load_beam_combiner(self.beam_combiner)

    @field_validator('amplitude_perturbation_lower_limit')
    def _validate_amplitude_perturbation_lower_limit(cls, value: Any, info: ValidationInfo) -> Tensor:
        """Validate the amplitude perturbation lower limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The amplitude perturbation lower limit
        """
        if value < 0:
            raise ValueError(f'{value} is not a valid input for {info.field_name}. Can not be negative.')
        return torch.tensor(value, dtype=torch.float32)

    @field_validator('amplitude_perturbation_upper_limit')
    def _validate_amplitude_perturbation_upper_limit(cls, value: Any, info: ValidationInfo) -> Tensor:
        """Validate the amplitude perturbation upper limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The amplitude perturbation upper limit
        """
        if value < 0:
            raise ValueError(f'{value} is not a valid input for {info.field_name}. Can not be negative.')
        if value <= info.data['amplitude_perturbation_lower_limit']:
            raise ValueError(f'{value} is not a valid input for {info.field_name}. Must be greater than '
                             f'{info.data["amplitude_perturbation_lower_limit"]}.')
        return torch.tensor(value, dtype=torch.float32)

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

    @field_validator('phase_perturbation_rms')
    def _validate_phase_perturbation_rms(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the phase perturbation rms input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The phase perturbation rms in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('polarization_perturbation_rms')
    def _validate_polarization_perturbation_rms(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the polarization perturbation rms input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The polarization perturbation rms in units of radians
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,)).si.value

    @field_validator('unperturbed_instrument_throughput')
    def _validate_unperturbed_instrument_throughput(cls, value: Any, info: ValidationInfo) -> Tensor:
        return torch.tensor(value, dtype=torch.float32)

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
        return self._calculate_wavelength_bins()

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

    def _calculate_wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def _load_array(self, array_configuration_type) -> Array:
        """Return the array configuration object from the dictionary.

        :param config_dict: The dictionary
        :return: The array configuration object.
        """

        match array_configuration_type:
            case ArrayEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation()

            case ArrayEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch()

            case ArrayEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation()

            case ArrayEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation()

    def _load_beam_combiner(self, beam_combination_scheme_type) -> BeamCombiner:
        """Return the beam combination scheme object from the dictionary.

        :param beam_combination_scheme_type: The beam combination scheme type
        :return: The beam combination object.
        """

        match beam_combination_scheme_type:
            case BeamCombinerEnum.DOUBLE_BRACEWELL.value:
                return DoubleBracewell()

            case BeamCombinerEnum.KERNEL_3.value:
                return Kernel3()

            case BeamCombinerEnum.KERNEL_4.value:
                return Kernel4()

            case BeamCombinerEnum.KERNEL_5.value:
                return Kernel5()

from functools import cached_property
from typing import Tuple, Union, Any

import astropy
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from src.sygn.core.entities.base_component import BaseComponent
from src.sygn.core.entities.observatory.array_configuration import (
    ArrayConfiguration,
    ArrayConfigurationEnum,
    EmmaXCircularRotation,
    EmmaXDoubleStretch,
    EquilateralTriangleCircularRotation,
    RegularPentagonCircularRotation,
)
from src.sygn.core.entities.observatory.beam_combination_scheme import (
    BeamCombinationScheme,
    BeamCombinationSchemeEnum,
    DoubleBracewell,
    Kernel3,
    Kernel4,
    Kernel5,
)
from src.sygn.core.entities.photon_sources.star import Star
from src.sygn.core.util.noise_generator import get_perturbation_time_series
from src.sygn.io.validators import validate_quantity_units


class Observatory(BaseComponent, BaseModel):
    """Class representing the observatory."""

    array_configuration: str
    beam_combination_scheme: str
    aperture_diameter: str
    spectral_resolving_power: int
    wavelength_range_lower_limit: str
    wavelength_range_upper_limit: str
    unperturbed_instrument_throughput: float
    amplitude_perturbation_rms: float
    amplitude_falloff_exponent: float
    phase_perturbation_rms: str
    phase_falloff_exponent: float
    polarization_perturbation_rms: str
    polarization_falloff_exponent: float
    field_of_view: Any = None
    amplitude_perturbation_time_series: Any = None
    phase_perturbation_time_series: Any = None
    polarization_perturbation_time_series: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.array_configuration = self._load_array_configuration(self.array_configuration)
        self.beam_combination_scheme = self._load_beam_combination_scheme(self.beam_combination_scheme)

    @field_validator('aperture_diameter')
    def _validate_aperture_diameter(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the aperture diameter input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The aperture diameter in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).to(u.m)

    @field_validator('phase_perturbation_rms')
    def _validate_phase_perturbation_rms(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the phase perturbation rms input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The phase perturbation rms in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('polarization_perturbation_rms')
    def _validate_polarization_perturbation_rms(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the polarization perturbation rms input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The polarization perturbation rms in units of radians
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,))

    @field_validator('wavelength_range_lower_limit')
    def _validate_wavelength_range_lower_limit(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the wavelength range lower limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The lower wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).to(u.um)

    @field_validator('wavelength_range_upper_limit')
    def _validate_wavelength_range_upper_limit(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the wavelength range upper limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The upper wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).to(u.um)

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

    def _calculate_amplitude_perturbation_time_series(self, settings) -> np.ndarray:
        return np.random.uniform(0.8, 0.9, (self.beam_combination_scheme.number_of_inputs, len(settings.time_steps))) \
            if settings.has_amplitude_perturbations else np.ones(
            (self.beam_combination_scheme.number_of_inputs, len(settings.time_steps)))

    def _calculate_phase_perturbation_time_series(self, settings, observation) -> np.ndarray:
        return get_perturbation_time_series(
            self.beam_combination_scheme.number_of_inputs,
            observation.exposure_time,
            len(settings.time_steps),
            self.phase_perturbation_rms,
            self.phase_falloff_exponent
        ) \
            if settings.has_phase_perturbations else np.zeros(
            (self.beam_combination_scheme.number_of_inputs, len(settings.time_steps)))

    def _calculate_polarization_perturbation_time_series(self, settings, observation) -> np.ndarray:
        return get_perturbation_time_series(
            self.beam_combination_scheme.number_of_inputs,
            observation.exposure_time,
            len(settings.time_steps),
            self.polarization_perturbation_rms,
            self.polarization_falloff_exponent
        ) \
            if settings.has_polarization_perturbations else np.zeros(
            (self.beam_combination_scheme.number_of_inputs, len(settings.time_steps)))

    def _calculate_wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the wavelength bin centers and widths. The wavelength bin widths are calculated starting from the
        wavelength lower range. As a consequence, the uppermost wavelength bin might be smaller than anticipated.

        :return: A tuple containing the wavelength bin centers and widths
        """
        current_minimum_wavelength = self.wavelength_range_lower_limit.value
        wavelength_bin_centers = []
        wavelength_bin_widths = []

        while current_minimum_wavelength <= self.wavelength_range_upper_limit.value:
            center_wavelength = current_minimum_wavelength / (1 - 1 / (2 * self.spectral_resolving_power))
            bin_width = 2 * (center_wavelength - current_minimum_wavelength)
            if (center_wavelength + bin_width / 2 <= self.wavelength_range_upper_limit.value):
                wavelength_bin_centers.append(center_wavelength)
                wavelength_bin_widths.append(bin_width)
                current_minimum_wavelength = center_wavelength + bin_width / 2
            else:
                last_bin_width = self.wavelength_range_upper_limit.value - current_minimum_wavelength
                last_center_wavelength = self.wavelength_range_upper_limit.value - last_bin_width / 2
                wavelength_bin_centers.append(last_center_wavelength)
                wavelength_bin_widths.append(last_bin_width)
                break
        return np.array(wavelength_bin_centers) * u.um, np.array(wavelength_bin_widths) * u.um,

    def _get_optimal_baseline(self,
                              optimized_differential_output: int,
                              optimized_wavelength: astropy.units.Quantity,
                              optimized_angular_distance: astropy.units.Quantity):
        factors = (1,)
        match (self.array_configuration.type, self.beam_combination_scheme.type):

            # 3 collector arrays
            case (ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.KERNEL_3.value, ):
                factors = (0.67,)

            # 4 collector arrays
            case (ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value, ):
                factors = (0.6,)
            case (ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.KERNEL_4.value, ):
                factors = 0.31, 1, 0.6
                print(
                    "The optimal baseline for Emma-X with kernel nulling is ill-defined for second differential output.")
            case (ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value,
                  BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value):
                factors = (1,)
                raise Warning("The optimal baseline for Emma-X with double stretching is not yet implemented.")
            case (ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value,
                  BeamCombinationSchemeEnum.KERNEL_4.value):
                factors = 1, 1, 1
                raise Warning("The optimal baseline for Emma-X with double stretching is not yet implemented."
                              )
            # 5 collector arrays
            case (ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.KERNEL_5.value):
                factors = 1.04, 0.67

        return factors[optimized_differential_output] * optimized_wavelength.to(u.m) / optimized_angular_distance.to(
            u.rad) * u.rad

    def _load_array_configuration(self, array_configuration_type) -> ArrayConfiguration:
        """Return the array configuration object from the dictionary.

        :param config_dict: The dictionary
        :return: The array configuration object.
        """

        match array_configuration_type:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation()

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch()

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation()

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation()

    def _load_beam_combination_scheme(self, beam_combination_scheme_type) -> BeamCombinationScheme:
        """Return the beam combination scheme object from the dictionary.

        :param config_dict: The dictionary
        :return: The beam combination object.
        """

        match beam_combination_scheme_type:
            case BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value:
                return DoubleBracewell()

            case BeamCombinationSchemeEnum.KERNEL_3.value:
                return Kernel3()

            case BeamCombinationSchemeEnum.KERNEL_4.value:
                return Kernel4()

            case BeamCombinationSchemeEnum.KERNEL_5.value:
                return Kernel5()

    def prepare(self, settings, observation):
        """Prepare the observatory for the simulation."""
        self.field_of_view = settings.wavelength_steps.to(u.m) / self.aperture_diameter * u.rad

        self.amplitude_perturbation_time_series = self._calculate_amplitude_perturbation_time_series(settings)

        self.phase_perturbation_time_series = self._calculate_phase_perturbation_time_series(settings, observation)

        self.polarization_perturbation_time_series = self._calculate_polarization_perturbation_time_series(settings,
                                                                                                           observation)

    def set_optimal_baseline(self,
                             star: Star,
                             optimized_differential_output: int,
                             optimized_wavelength: astropy.units.Quantity,
                             optimized_star_separation: Union[str, astropy.units.Quantity],
                             baseline_minimum: astropy.units.Quantity,
                             baseline_maximum: astropy.units.Quantity):
        """Set the baseline to optimize for the habitable zone, if it is between the minimum and maximum allowed
        baselines.

        :param star: The star object
        :param optimized_differential_output: The optimized differential output index
        :param optimized_wavelength: The optimized wavelength
        :param optimzied_star_separation: The angular radius of the habitable zone
        """
        # Get the optimized separation in angular units, if it is not yet in angular units
        if optimized_star_separation == "habitable-zone":
            optimized_star_separation = star.habitable_zone_central_angular_radius
        elif optimized_star_separation.unit.is_equivalent(u.m):
            optimized_star_separation = (
                    optimized_star_separation.to(u.m) / star.distance.to(u.m) * u.rad
            )

        # Get the optimal baseline and check if it is within the allowed range
        optimal_baseline = self._get_optimal_baseline(
            optimized_differential_output=optimized_differential_output,
            optimized_wavelength=optimized_wavelength,
            optimized_angular_distance=optimized_star_separation,
        ).to(u.m)

        if (
                baseline_minimum.to(u.m).value <= optimal_baseline.value
                and optimal_baseline.value <= baseline_maximum.to(u.m).value
        ):
            self.array_configuration.baseline = optimal_baseline
        else:
            raise ValueError(
                f"Optimal baseline of {optimal_baseline} is not within allowed ranges of baselines {self.array_configuration.baseline_minimum}-{self.array_configuration.baseline_maximum}"
            )

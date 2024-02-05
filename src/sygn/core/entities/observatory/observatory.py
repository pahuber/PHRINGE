from typing import Tuple, Union

import astropy
import numpy as np
from astropy import units as u
from pydantic import BaseModel

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
    phase_rms: str
    phase_falloff_exponent: float
    polarization_perturbation_rms: str
    polarization_falloff_exponent: float

    def __init__(self, **data):
        super().__init__(**data)
        self.beam_combination_scheme = self._load_beam_combination_scheme(
            self.beam_combination_scheme
        )
        self.wavelength_bin_centers, self.wavelength_bin_widths = (
            self._calculate_wavelength_bins()
        )
        self.fields_of_view = self._calculate_fields_of_view()

    def _calculate_amplitude_perturbation_distributions(self, settings) -> np.ndarray:
        pass

    def _calculate_fields_of_view(self) -> np.ndarray:
        """Return the fields of view for each wavelength.

        :return: An array containing the field of view for each wavelength
        """
        return self.wavelength_bin_centers.to(u.m) / self.aperture_diameter * u.rad

    def _calculate_phase_perturbation_distributions(self, settings) -> np.ndarray:
        pass

    def _calculate_polarization_perturbation_distributions(
            self, settings
    ) -> np.ndarray:
        pass

    def _calculate_wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the wavelength bin centers and widths. The wavelength bin widths are calculated starting from the
        wavelength lower range. As a consequence, the uppermost wavelength bin might be smaller than anticipated.

        :return: A tuple containing the wavelength bin centers and widths
        """
        current_minimum_wavelength = self.wavelength_range_lower_limit.value
        wavelength_bin_centers = []
        wavelength_bin_widths = []

        while current_minimum_wavelength <= self.wavelength_range_upper_limit.value:
            center_wavelength = current_minimum_wavelength / (
                    1 - 1 / (2 * self.spectral_resolving_power)
            )
            bin_width = 2 * (center_wavelength - current_minimum_wavelength)
            if (
                    center_wavelength + bin_width / 2
                    <= self.wavelength_range_upper_limit.value
            ):
                wavelength_bin_centers.append(center_wavelength)
                wavelength_bin_widths.append(bin_width)
                current_minimum_wavelength = center_wavelength + bin_width / 2
            else:
                last_bin_width = (
                        self.wavelength_range_upper_limit.value - current_minimum_wavelength
                )
                last_center_wavelength = (
                        self.wavelength_range_upper_limit.value - last_bin_width / 2
                )
                wavelength_bin_centers.append(last_center_wavelength)
                wavelength_bin_widths.append(last_bin_width)
                break
        return (
            np.array(wavelength_bin_centers) * u.um,
            np.array(wavelength_bin_widths) * u.um,
        )

    def _get_optimal_baseline(
            self,
            optimized_differential_output: int,
            optimized_wavelength: astropy.units.Quantity,
            optimized_angular_distance: astropy.units.Quantity,
    ):
        factors = (1,)
        match (self.array_configuration.type, self.beam_combination_scheme.type):
            # 3 collector arrays
            case (
                ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value,
                BeamCombinationSchemeEnum.KERNEL_3.value,
            ):
                factors = (0.67,)
            # 4 collector arrays
            case (
                ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
                BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value,
            ):
                factors = (0.6,)
            case (
                ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
                BeamCombinationSchemeEnum.KERNEL_4.value,
            ):
                factors = 0.31, 1, 0.6
                print(
                    "The optimal baseline for Emma-X with kernel nulling is ill-defined for second differential output."
                )
            case (
                ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value,
                BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value,
            ):
                factors = (1,)
                raise Warning(
                    "The optimal baseline for Emma-X with double stretching is not yet implemented."
                )
            case (
                ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value,
                BeamCombinationSchemeEnum.KERNEL_4.value,
            ):
                factors = 1, 1, 1
                raise Warning(
                    "The optimal baseline for Emma-X with double stretching is not yet implemented."
                )
            # 5 collector arrays
            case (
                ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value,
                BeamCombinationSchemeEnum.KERNEL_5.value,
            ):
                factors = 1.04, 0.67
        return (
                factors[optimized_differential_output]
                * optimized_wavelength.to(u.m)
                / optimized_angular_distance.to(u.rad)
                * u.rad
        )

    def _load_array_configuration(
            self, array_configuration_type, modulation_period, baseline_ratio
    ) -> ArrayConfiguration:
        """Return the array configuration object from the dictionary.

        :param config_dict: The dictionary
        :return: The array configuration object.
        """

        match array_configuration_type:
            case ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value:
                return EmmaXCircularRotation(
                    modulation_period=modulation_period, baseline_ratio=baseline_ratio
                )

            case ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value:
                return EmmaXDoubleStretch(
                    modulation_period=modulation_period, baseline_ratio=baseline_ratio
                )

            case ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value:
                return EquilateralTriangleCircularRotation(
                    modulation_period=modulation_period, baseline_ratio=baseline_ratio
                )

            case ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value:
                return RegularPentagonCircularRotation(
                    modulation_period=modulation_period, baseline_ratio=baseline_ratio
                )

    def _load_beam_combination_scheme(
            self, beam_combination_scheme_type
    ) -> BeamCombinationScheme:
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
        self.array_configuration = self._load_array_configuration(
            self.array_configuration,
            observation.modulation_period,
            observation.baseline_ratio,
        )
        self.amplitude_perturbation_distributions = (
            self._calculate_amplitude_perturbation_distributions(settings)
        )
        self.phase_perterbation_distributions = (
            self._calculate_phase_perturbation_distributions(settings)
        )
        self.polarization_perturbation_distributions = (
            self._calculate_polarization_perturbation_distributions(settings)
        )

    def set_optimal_baseline(
            self,
            star: Star,
            optimized_differential_output: int,
            optimized_wavelength: astropy.units.Quantity,
            optimized_star_separation: Union[str, astropy.units.Quantity],
            baseline_minimum: astropy.units.Quantity,
            baseline_maximum: astropy.units.Quantity,
    ):
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

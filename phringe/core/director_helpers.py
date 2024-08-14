from typing import Union

import torch
from torch import Tensor

from phringe.core.entities.observatory.array import ArrayEnum
from phringe.core.entities.observatory.beam_combiner import BeamCombinerEnum
from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.photon_sources.star import Star
from phringe.util.helpers import InputSpectrum
from phringe.util.noise_generator import get_perturbation_time_series


def calculate_amplitude_perturbations(
        number_of_inputs: int,
        simulation_time_steps: Tensor,
        amplitude_perturbation_lower_limit: float,
        amplitude_perturbation_upper_limit: float,
        has_amplitude_perturbations: bool
) -> Tensor:
    """Return the amplitude perturbation time series in units of 1 as an array of shape N_collectors x N_time_steps.

    :param number_of_inputs: The number of inputs
    :param simulation_time_steps: The simulation time steps
    :param amplitude_perturbation_lower_limit: The amplitude perturbation lower limit
    :param amplitude_perturbation_upper_limit: The amplitude perturbation upper limit
    :param has_amplitude_perturbations: Whether the simulation has amplitude perturbations
    :return: The amplitude perturbation time series in units of 1
    """
    # TODO: Update this
    return amplitude_perturbation_lower_limit + (
            amplitude_perturbation_upper_limit - amplitude_perturbation_lower_limit) * torch.rand(
        (number_of_inputs, len(simulation_time_steps)),
        dtype=torch.float32) if has_amplitude_perturbations else torch.ones(
        (number_of_inputs, len(simulation_time_steps)))


def calculate_nulling_baseline(
        star_habitable_zone_central_angular_radius: float,
        star_distance: float,
        optimized_star_separation: Union[str, float],
        optimized_differential_output: int,
        optimized_wavelength: float,
        baseline_maximum: float,
        baseline_minimum: float,
        array_configuration_type: str,
        beam_combination_scheme_type: str
) -> float:
    """Calculate the nulling baseline in meters.

    :param star_habitable_zone_central_angular_radius: The star habitable zone central angular radius
    :param star_distance: The star distance
    :param optimized_differential_output: The optimized differential output
    :param optimized_wavelength: The optimized wavelength
    :param optimized_star_separation: The optimized star separation
    :param baseline_maximum: The baseline maximum
    :param baseline_minimum: The baseline minimum
    :param array_configuration_type: The array configuration type
    :param beam_combination_scheme_type: The beam combination scheme type
    :return: The nulling baseline in meters
    """
    # Get the optimized separation in angular units, if it is not yet in angular units
    if optimized_star_separation == "habitable-zone":
        optimized_star_separation = star_habitable_zone_central_angular_radius

    # Get the optimal baseline and check if it is within the allowed range
    # TODO: Check all factors again
    factors = (1,)
    match (array_configuration_type, beam_combination_scheme_type):

        # 3 collector arrays
        case (
            ArrayEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value,
            BeamCombinerEnum.KERNEL_3.value
        ):
            factors = (0.67,)

        # 4 collector arrays
        case (
            ArrayEnum.EMMA_X_CIRCULAR_ROTATION.value,
            BeamCombinerEnum.DOUBLE_BRACEWELL.value
        ):
            factors = (0.6,)

        case (
            ArrayEnum.EMMA_X_CIRCULAR_ROTATION.value,
            BeamCombinerEnum.KERNEL_4.value
        ):
            factors = 0.31, 1, 0.6
            print(
                "The optimal baseline for Emma-X with kernel nulling is ill-defined for second differential output.")

        case (
            ArrayEnum.EMMA_X_DOUBLE_STRETCH.value,
            BeamCombinerEnum.DOUBLE_BRACEWELL.value
        ):
            factors = (1,)
            raise Warning("The optimal baseline for Emma-X with double stretching is not yet implemented.")

        case (
            ArrayEnum.EMMA_X_DOUBLE_STRETCH.value,
            BeamCombinerEnum.KERNEL_4.value
        ):
            factors = 1, 1, 1
            raise Warning("The optimal baseline for Emma-X with double stretching is not yet implemented."
                          )
        # 5 collector arrays
        case (
            ArrayEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value,
            BeamCombinerEnum.KERNEL_5.value
        ):
            factors = 1.04, 0.67

    nulling_baseline = factors[optimized_differential_output] * optimized_wavelength / optimized_star_separation

    if baseline_minimum <= nulling_baseline and nulling_baseline <= baseline_maximum:
        return nulling_baseline
    raise ValueError(
        f"Nulling baseline of {nulling_baseline} is not within allowed ranges of baselines {baseline_minimum}-{baseline_maximum}"
    )


def calculate_phase_perturbations(
        number_of_inputs: int,
        detector_integration_time: float,
        simulation_time_steps: Tensor,
        phase_perturbation_rms: float,
        phase_falloff_exponent: float,
        has_phase_perturbations: bool
) -> Tensor:
    """Return the phase perturbation time series in units of meters  as an array of shape N_collectors x N_time_steps.

    :param settings: The settings object
    :param observation: The observation object
    :return: The phase perturbation time series in units of meters
    """
    return get_perturbation_time_series(
        number_of_inputs,
        detector_integration_time,
        len(simulation_time_steps),
        phase_perturbation_rms,
        phase_falloff_exponent
    ) if has_phase_perturbations else torch.zeros((number_of_inputs, len(simulation_time_steps)), dtype=torch.float32)


def calculate_polarization_perturbations(
        number_of_inputs: int,
        detector_integration_time: float,
        simulation_time_steps: Tensor,
        polarization_perturbation_rms: float,
        polarization_falloff_exponent: float,
        has_polarization_perturbations: bool
) -> Tensor:
    """Return the polarization perturbation time series in units of rad  as an array of shape N_collectors x N_time_steps.

    :param settings: The settings object
    :param observation: The observation object
    :return: The polarization perturbation time series in units of rad
    """
    return get_perturbation_time_series(
        number_of_inputs,
        detector_integration_time,
        len(simulation_time_steps),
        polarization_perturbation_rms,
        polarization_falloff_exponent
    ) if has_polarization_perturbations else torch.zeros(
        (number_of_inputs, len(simulation_time_steps)),
        dtype=torch.float32
    )


def calculate_simulation_time_steps(total_integration_time: float, simulation_time_step_length: float) -> Tensor:
    """Calculate the simulation time steps in seconds.

    :param total_integration_time: The total integration time in seconds
    :return: The simulation time steps in seconds
    """
    number_of_steps = int(total_integration_time / simulation_time_step_length)
    return torch.linspace(0, total_integration_time, number_of_steps)


def prepare_modeled_sources(
        sources: list[BasePhotonSource],
        simulation_time_steps: Tensor,
        simulation_wavelength_bin_centers: Tensor,
        input_spectra: list[InputSpectrum],
        grid_size: int,
        field_of_view: Tensor,
        solar_ecliptic_latitude: float,
        has_planet_orbital_motion: bool,
        has_planet_signal: bool,
        has_stellar_leakage: bool,
        has_local_zodi_leakage: bool,
        has_exozodi_leakage: bool
) -> list[BasePhotonSource]:
    """Return the spectral flux densities, brightness distributions and coordinates for all sources in the scene.

    :param sources: The sources in the scene
    :param simulation_time_steps: The simulation time steps
    :param simulation_wavelength_bin_centers: The simulation wavelength bin centers
    :param grid_size: The grid size
    :param field_of_view: The field of view
    :param solar_ecliptic_latitude: The solar ecliptic latitude
    :param has_planet_orbital_motion: Whether the simulation has planet orbital motion
    :param has_stellar_leakage: Whether the simulation has stellar leakage
    :param has_local_zodi_leakage: Whether the simulation has local zodi leakage
    :param has_exozodi_leakage: Whether the simulation has exozodi leakage
    :return: The prepared sources
    """
    star = [star for star in sources if isinstance(star, Star)][0]
    planets = [planet for planet in sources if isinstance(planet, Planet)]
    local_zodi = [local_zodi for local_zodi in sources if isinstance(local_zodi, LocalZodi)][0]
    exozodi = [exozodi for exozodi in sources if isinstance(exozodi, Exozodi)][0]
    prepared_sources = []

    if has_planet_signal:
        for index_planet, planet in enumerate(planets):
            planet.prepare(
                simulation_wavelength_bin_centers,
                grid_size,
                star_distance=star.distance,
                input_spectra=input_spectra,
                time_steps=simulation_time_steps,
                has_planet_orbital_motion=has_planet_orbital_motion,
                star_mass=star.mass,
                number_of_wavelength_steps=len(simulation_wavelength_bin_centers)
            )
            prepared_sources.append(planet)
    if has_stellar_leakage:
        star.prepare(
            simulation_wavelength_bin_centers,
            grid_size,
            number_of_wavelength_steps=len(simulation_wavelength_bin_centers)
        )
        prepared_sources.append(star)
    if has_local_zodi_leakage:
        local_zodi.prepare(
            simulation_wavelength_bin_centers,
            grid_size,
            field_of_view=field_of_view,
            star_right_ascension=star.right_ascension,
            star_declination=star.declination,
            solar_ecliptic_latitude=solar_ecliptic_latitude,
            number_of_wavelength_steps=len(simulation_wavelength_bin_centers)
        )
        prepared_sources.append(local_zodi)
    if has_exozodi_leakage:
        exozodi.prepare(
            simulation_wavelength_bin_centers,
            grid_size,
            field_of_view=field_of_view,
            star_distance=star.distance,
            star_luminosity=star.luminosity)
        prepared_sources.append(exozodi)

    return prepared_sources

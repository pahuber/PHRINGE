from typing import Union

import spectres
import torch
from torch import Tensor

from phringe.core.entities.observatory.array_configuration import ArrayConfigurationEnum
from phringe.core.entities.observatory.beam_combination_scheme import BeamCombinationSchemeEnum
from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.photon_sources.star import Star
from phringe.util.helpers import InputSpectrum
from phringe.util.noise_generator import get_perturbation_time_series
from phringe.util.spectrum import create_blackbody_spectrum


def calculate_amplitude_perturbations(
        number_of_inputs: int,
        simulation_time_steps: Tensor,
        has_amplitude_perturbations: bool
) -> Tensor:
    """Return the amplitude perturbation time series in units of 1.

    :param number_of_inputs: The number of inputs
    :param simulation_time_steps: The simulation time steps
    :param has_amplitude_perturbations: Whether the simulation has amplitude perturbations
    :return: The amplitude perturbation time series in units of 1
    """
    # TODO: Update this
    return 0.7 + (0.9 - 0.7) * torch.rand((
        number_of_inputs,
        len(simulation_time_steps)), dtype=torch.float32) \
        if has_amplitude_perturbations else torch.ones(
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
            ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value,
            BeamCombinationSchemeEnum.KERNEL_3.value
        ):
            factors = (0.67,)

        # 4 collector arrays
        case (
            ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
            BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value
        ):
            factors = (0.6,)

        case (
            ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
            BeamCombinationSchemeEnum.KERNEL_4.value
        ):
            factors = 0.31, 1, 0.6
            print(
                "The optimal baseline for Emma-X with kernel nulling is ill-defined for second differential output.")

        case (
            ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value,
            BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value
        ):
            factors = (1,)
            raise Warning("The optimal baseline for Emma-X with double stretching is not yet implemented.")

        case (
            ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value,
            BeamCombinationSchemeEnum.KERNEL_4.value
        ):
            factors = 1, 1, 1
            raise Warning("The optimal baseline for Emma-X with double stretching is not yet implemented."
                          )
        # 5 collector arrays
        case (
            ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value,
            BeamCombinationSchemeEnum.KERNEL_5.value
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
    """Return the phase perturbation time series in units of meters.

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
    """Return the polarization perturbation time series in units of rad.

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


def calculate_simulation_wavelength_bin_centers(
        instrument_wavelength_bin_centers: Tensor,
        maximum_simulation_wavelength_sampling: int,
        wavelength_range_lower_limit: float,
        wavelength_range_upper_limit: float,
        reference_spectra: list[Tensor]
) -> Tensor:
    """Calculate the optimized wavelength sampling for the simulation. This is done by taking the gradient of the
    normalized planet spectra and adding extra wavelength steps (to the instrument wavelength bins) where the
    gradient is larger than the threshold. This assures a good sampling of the planet spectra if the instrument spectral
    resolving power is low compared to the variation of the spectra.

    :param instrument_wavelength_bin_centers: The instrument wavelength bin centers
    :param maximum_simulation_wavelength_sampling: The maximum resolution of the simulation wavelength range
    :param wavelength_range_lower_limit: The lower limit of the wavelength range
    :param wavelength_range_upper_limit: The upper limit of the wavelength range
    :param reference_spectra: The reference spectra
    :return: The wavelength bin centers in units of meters
    """
    # Generate the maximum resolution wavelength range. Spectra will never be sampled higher than this resolution
    wavelength_range = torch.linspace(
        wavelength_range_lower_limit,
        wavelength_range_upper_limit,
        maximum_simulation_wavelength_sampling
    )
    optimum_wavelength_bin_centers = []

    for reference_spectrum in reference_spectra:

        # Calculate gradient of normalized reference planet spectrum
        spectrum_gradient = torch.gradient(reference_spectrum / torch.max(reference_spectrum))[0]

        # Create a mask of ones where the gradient is larger than the threshold and zeros elsewhere
        gradient_threshold = 0.05
        indices = torch.where(torch.abs(spectrum_gradient) > gradient_threshold)
        mask = torch.zeros(len(wavelength_range))
        mask[indices] = 1

        for index, value in enumerate(mask):

            # Add the wavelength step to the list of optimum wavelength steps if the mask value is 1
            if value == 1:
                optimum_wavelength_bin_centers.append(wavelength_range[index])

        # Update the optimum wavelength bin centers list
        optimum_wavelength_bin_centers = sorted(
            optimum_wavelength_bin_centers + list(instrument_wavelength_bin_centers)
        )

    return torch.unique(torch.tensor(optimum_wavelength_bin_centers, dtype=torch.float32))


def calculate_simulation_wavelength_bin_widths(
        wavelength_range_lower_limit: float,
        wavelength_range_upper_limit: float,
        simulation_wavelength_bin_centers: Tensor
) -> Tensor:
    """Calculate the simulation wavelength bin widths from the simulation bin centers, in units of meters.

    :param wavelength_range_lower_limit: The lower limit of the wavelength range
    :param wavelength_range_upper_limit: The upper limit of the wavelength range
    :param simulation_wavelength_bin_centers: The simulation wavelength bin centers
    :return: The simulation wavelength bin widths in units of meters
    """
    current_edge = wavelength_range_lower_limit
    bin_widths = []

    for index, wavelength in enumerate(simulation_wavelength_bin_centers):
        upper_wavelength = simulation_wavelength_bin_centers[index + 1] \
            if index < len(simulation_wavelength_bin_centers) - 1 else wavelength_range_upper_limit
        bin_widths.append(
            ((wavelength - current_edge) + (upper_wavelength - wavelength) / 2))
        current_edge += bin_widths[index]
    return torch.asarray(bin_widths, dtype=torch.float32)


def calculate_simulation_wavelength_bins(
        wavelength_range_lower_limit: float,
        wavelength_range_upper_limit: float,
        maximum_simulation_wavelength_sampling: int,
        instrument_wavelength_bin_centers: Tensor,
        planets: list[Planet],
        input_spectra: list[InputSpectrum]
) -> tuple[Tensor, Tensor]:
    reference_spectra = generate_reference_spectra(
        wavelength_range_lower_limit,
        wavelength_range_upper_limit,
        maximum_simulation_wavelength_sampling,
        planets,
        input_spectra
    )
    simulation_wavelength_bin_centers = calculate_simulation_wavelength_bin_centers(
        instrument_wavelength_bin_centers,
        maximum_simulation_wavelength_sampling,
        wavelength_range_lower_limit,
        wavelength_range_upper_limit,
        reference_spectra
    )
    simulation_wavelength_bin_widths = calculate_simulation_wavelength_bin_widths(
        wavelength_range_lower_limit,
        wavelength_range_upper_limit,
        simulation_wavelength_bin_centers
    )
    return simulation_wavelength_bin_centers, simulation_wavelength_bin_widths, reference_spectra


def generate_reference_spectra(
        wavelength_range_lower_limit: float,
        wavelength_range_upper_limit: float,
        maximum_simulation_wavelength_sampling: int,
        planets: list[Planet],
        input_spectra: list[InputSpectrum]
) -> list[Tensor]:
    """Generate reference spectra for all planets in the scene in units of ph m-3 s-1 sr-1. If no input spectrum is
    provided for a planet, generate a blackbody spectrum for it.

    :param wavelength_range_lower_limit: The lower limit of the wavelength range
    :param wavelength_range_upper_limit: The upper limit of the wavelength range
    :param maximum_simulation_wavelength_sampling: The resolution of the simulation wavelength range
    :param planets: The planets in the scene
    :param input_spectra: The input spectra
    :return: The reference spectra in units of ph m-3 s-1 sr-1
    """
    # Generate the maximum resolution wavelength range. Spectra will never be sampled higher than this resolution
    wavelength_range = torch.linspace(
        wavelength_range_lower_limit,
        wavelength_range_upper_limit,
        maximum_simulation_wavelength_sampling
    )
    input_spectra_planet_names = [spectrum.planet_name for spectrum in input_spectra] if input_spectra else []
    reference_spectra = []
    for planet in planets:
        if planet.name in input_spectra_planet_names:
            input_spectrum = next(
                (input_spectrum for input_spectrum in input_spectra if input_spectrum.planet_name == planet.name),
                None
            )
            reference_spectrum = spectres.spectres(
                wavelength_range.numpy(),
                input_spectrum.wavelengths.numpy(),
                input_spectrum.spectral_flux_density.numpy(),
                fill=0,
                verbose=False)
            reference_spectra.append(torch.asarray(reference_spectrum, dtype=torch.float32))
        else:
            reference_spectra.append(
                torch.asarray(
                    create_blackbody_spectrum(
                        planet.temperature,
                        wavelength_range
                    )
                    , dtype=torch.float32)
            )
    return reference_spectra


def prepare_modeled_sources(
        sources: list[BasePhotonSource],
        simulation_time_steps: Tensor,
        simulation_wavelength_bin_centers: Tensor,
        wavelength_range_lower_limit: float,
        wavelength_range_upper_limit: float,
        maximum_simulation_wavelength_sampling: int,
        reference_spectra: list[Tensor],
        grid_size: int,
        field_of_view: Tensor,
        solar_ecliptic_latitude: float,
        has_planet_orbital_motion: bool,
        has_stellar_leakage: bool,
        has_local_zodi_leakage: bool,
        has_exozodi_leakage: bool
) -> list[BasePhotonSource]:
    wavelength_range = torch.linspace(
        wavelength_range_lower_limit,
        wavelength_range_upper_limit,
        maximum_simulation_wavelength_sampling
    )
    star = [star for star in sources if isinstance(star, Star)][0]
    planets = [planet for planet in sources if isinstance(planet, Planet)]
    local_zodi = [local_zodi for local_zodi in sources if isinstance(local_zodi, LocalZodi)][0]
    exozodi = [exozodi for exozodi in sources if isinstance(exozodi, Exozodi)][0]
    prepared_sources = []

    for index_planet, planet in enumerate(planets):
        planet.prepare(
            simulation_wavelength_bin_centers,
            grid_size,
            star_distance=star.distance,
            reference_spectrum=reference_spectra[index_planet],
            reference_wavelength_bin_centers=wavelength_range,
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

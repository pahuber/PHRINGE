import torch
from torch import tensor

from phringe.core.director_helpers import calculate_simulation_time_steps, calculate_nulling_baseline, \
    calculate_simulation_wavelength_bins, calculate_amplitude_perturbations, calculate_phase_perturbations, \
    calculate_polarization_perturbations, prepare_sources
from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.util.helpers import InputSpectrum


class Director():
    """Class representation of the director."""
    simulation_time_step_length = tensor(60, dtype=torch.uint8)
    maximum_simulation_wavelength_sampling = 1000

    def __init__(
            self,
            settings: Settings,
            observatory: Observatory,
            observation: Observation,
            scene: Scene,
            input_spectra: list[InputSpectrum]
    ):
        self._settings = settings
        self._observatory = observatory
        self._observation = observation
        self._scene = scene
        self._input_spectra = input_spectra

    def run(self):
        # Calculate simulation time steps
        simulation_time_steps = calculate_simulation_time_steps(
            self._observation.total_integration_time,
            self.simulation_time_step_length
        )

        # Calculate the simulation wavelength bins
        simulation_wavelength_bin_centers, simulation_wavelength_bin_widths, reference_spectra = (
            calculate_simulation_wavelength_bins(
                self._observatory.wavelength_range_lower_limit,
                self._observatory.wavelength_range_upper_limit,
                self.maximum_simulation_wavelength_sampling,
                self._observatory.wavelength_bin_centers,
                self._scene.planets,
                self._input_spectra
            )
        )

        # Calculate field of view
        field_of_view = simulation_wavelength_bin_centers / self._observatory.aperture_diameter

        # Calculate the nulling baseline
        nulling_baseline = calculate_nulling_baseline(
            self._scene.star.habitable_zone_central_angular_radius,
            self._scene.star.distance,
            self._observation.optimized_star_separation,
            self._observation.optimized_differential_output,
            self._observation.optimized_wavelength,
            self._observation.baseline_maximum,
            self._observation.baseline_minimum,
            self._observatory.array_configuration.type.value,
            self._observatory.beam_combination_scheme.type
        )

        # Calculate the instrument perturbations
        amplitude_perturbations = calculate_amplitude_perturbations(
            self._observatory.beam_combination_scheme.number_of_inputs,
            simulation_time_steps,
            self._settings.has_amplitude_perturbations
        )
        phase_perturbation_time_series = calculate_phase_perturbations(
            self._observatory.beam_combination_scheme.number_of_inputs,
            self._observation.detector_integration_time,
            simulation_time_steps,
            self._observatory.phase_perturbation_rms,
            self._observatory.phase_falloff_exponent,
            self._settings.has_phase_perturbations
        )
        polarization_perturbation_time_series = calculate_polarization_perturbations(
            self._observatory.beam_combination_scheme.number_of_inputs,
            self._observation.detector_integration_time,
            simulation_time_steps,
            self._observatory.polarization_perturbation_rms,
            self._observatory.polarization_falloff_exponent,
            self._settings.has_polarization_perturbations
        )

        # TODO: Estimate data size and start for loop, if memory is not sufficient

        # Calculate the observatory coordinates
        observatory_coordinates = self._observatory.array_configuration.get_collector_coordinates(
            simulation_time_steps,
            nulling_baseline,
            self._observation.modulation_period,
            self._observation.baseline_ratio
        )

        # Calculate the spectral flux densities, coordinates and sky brightness distributions of all sources
        self._scene = prepare_sources(
            self._scene,
            simulation_time_steps,
            simulation_wavelength_bin_centers,
            self._observatory.wavelength_range_lower_limit,
            self._observatory.wavelength_range_upper_limit,
            self.maximum_simulation_wavelength_sampling,
            reference_spectra,
            self._settings.grid_size,
            field_of_view,
            self._observation.solar_ecliptic_latitude,
            self._settings.has_planet_orbital_motion,
            self._settings.has_stellar_leakage,
            self._settings.has_local_zodi_leakage,
            self._settings.has_exozodi_leakage
        )

        # Generate the data
        a = 0

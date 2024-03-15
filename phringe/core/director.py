import torch
from torch import tensor

from phringe.core.data_generator import DataGenerator
from phringe.core.director_helpers import calculate_simulation_time_steps, calculate_nulling_baseline, \
    calculate_simulation_wavelength_bins, calculate_amplitude_perturbations, calculate_phase_perturbations, \
    calculate_polarization_perturbations, prepare_modeled_sources
from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.util.helpers import InputSpectrum


class Director():
    """Class representation of the director.

    :param device: The device
    :param aperture_diameter: The aperture diameter
    :param array_configuration: The array configuration
    :param baseline_maximum: The maximum baseline


    """
    _simulation_time_step_length = tensor(60, dtype=torch.uint8)
    _maximum_simulation_wavelength_sampling = 1000

    def __init__(
            self,
            settings: Settings,
            observatory: Observatory,
            observation: Observation,
            scene: Scene,
            input_spectra: list[InputSpectrum]
    ):
        """Constructor method.

        :param settings: The settings
        :param observatory: The observatory
        :param observation: The observation
        :param scene: The scene
        :param input_spectra: The input spectra
        """
        self._device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

        self._aperture_diameter = observatory.aperture_diameter
        self._array_configuration = observatory.array_configuration
        self._baseline_maximum = observation.baseline_maximum
        self._baseline_minimum = observation.baseline_minimum
        self._baseline_ratio = observation.baseline_ratio
        self._beam_combination_scheme = observatory.beam_combination_scheme
        self._beam_combination_transfer_matrix = self._beam_combination_scheme.get_beam_combination_transfer_matrix()
        self._detector_integration_time = observation.detector_integration_time
        self._differential_output_pairs = self._beam_combination_scheme.get_differential_output_pairs()
        self._grid_size = settings.grid_size
        self._has_amplitude_perturbations = settings.has_amplitude_perturbations
        self._has_exozodi_leakage = settings.has_exozodi_leakage
        self._has_local_zodi_leakage = settings.has_local_zodi_leakage
        self._has_phase_perturbations = settings.has_phase_perturbations
        self._has_planet_orbital_motion = settings.has_planet_orbital_motion
        self._has_polarization_perturbations = settings.has_polarization_perturbations
        self._has_stellar_leakage = settings.has_stellar_leakage
        self._input_spectra = input_spectra
        self._modulation_period = observation.modulation_period
        self._number_of_inputs = self._beam_combination_scheme.number_of_inputs
        self._number_of_outputs = self._beam_combination_scheme.number_of_outputs
        self._observatory_wavelength_bin_centers = observatory.wavelength_bin_centers
        self._observatory_wavelength_bin_edges = observatory.wavelength_bin_edges
        self._observatory_wavelength_bin_widths = observatory.wavelength_bin_widths
        self._observatory_wavelength_range_lower_limit = observatory.wavelength_range_lower_limit
        self._observatory_wavelength_range_upper_limit = observatory.wavelength_range_upper_limit
        self._optimized_differential_output = observation.optimized_differential_output
        self._optimized_star_separation = observation.optimized_star_separation
        self._optimized_wavelength = observation.optimized_wavelength
        self._phase_falloff_exponent = observatory.phase_falloff_exponent
        self._phase_perturbation_rms = observatory.phase_perturbation_rms
        self._planets = scene.planets
        self._polarization_falloff_exponent = observatory.polarization_falloff_exponent
        self._polarization_perturbation_rms = observatory.polarization_perturbation_rms
        self._solar_ecliptic_latitude = observation.solar_ecliptic_latitude
        self._sources = scene.get_all_sources()
        self._star = scene.star
        self._total_integration_time = observation.total_integration_time
        self._unperturbed_instrument_throughput = observatory.unperturbed_instrument_throughput

    def run(self):
        # Calculate simulation and instrument time steps
        simulation_time_steps = calculate_simulation_time_steps(
            self._total_integration_time,
            self._simulation_time_step_length
        )
        observatory_time_steps = torch.linspace(
            0,
            self._total_integration_time,
            int(self._total_integration_time / self._detector_integration_time)
        )

        # Calculate the simulation wavelength bins
        simulation_wavelength_bin_centers, simulation_wavelength_bin_widths, reference_spectra = (
            calculate_simulation_wavelength_bins(
                self._observatory_wavelength_range_lower_limit,
                self._observatory_wavelength_range_upper_limit,
                self._maximum_simulation_wavelength_sampling,
                self._observatory_wavelength_bin_centers,
                self._planets,
                self._input_spectra
            )
        )

        # Calculate field of view
        field_of_view = simulation_wavelength_bin_centers / self._aperture_diameter

        # Calculate the nulling baseline
        nulling_baseline = calculate_nulling_baseline(
            self._star.habitable_zone_central_angular_radius,
            self._star.distance,
            self._optimized_star_separation,
            self._optimized_differential_output,
            self._optimized_wavelength,
            self._baseline_maximum,
            self._baseline_minimum,
            self._array_configuration.type.value,
            self._beam_combination_scheme.type
        )

        # Calculate the instrument perturbations
        amplitude_perturbations = calculate_amplitude_perturbations(
            self._number_of_inputs,
            simulation_time_steps,
            self._has_amplitude_perturbations
        )
        phase_perturbations = calculate_phase_perturbations(
            self._number_of_inputs,
            self._detector_integration_time,
            simulation_time_steps,
            self._phase_perturbation_rms,
            self._phase_falloff_exponent,
            self._has_phase_perturbations
        )
        polarization_perturbations = calculate_polarization_perturbations(
            self._number_of_inputs,
            self._detector_integration_time,
            simulation_time_steps,
            self._polarization_perturbation_rms,
            self._polarization_falloff_exponent,
            self._has_polarization_perturbations
        )

        # Calculate the observatory coordinates
        observatory_coordinates = self._array_configuration.get_collector_coordinates(
            simulation_time_steps,
            nulling_baseline,
            self._modulation_period,
            self._baseline_ratio
        )

        # Calculate the spectral flux densities, coordinates and brightness distributions of all sources in the scene
        self._sources = prepare_modeled_sources(
            self._sources,
            simulation_time_steps,
            simulation_wavelength_bin_centers,
            self._observatory_wavelength_range_lower_limit,
            self._observatory_wavelength_range_upper_limit,
            self._maximum_simulation_wavelength_sampling,
            reference_spectra,
            self._grid_size,
            field_of_view,
            self._solar_ecliptic_latitude,
            self._has_planet_orbital_motion,
            self._has_stellar_leakage,
            self._has_local_zodi_leakage,
            self._has_exozodi_leakage
        )

        # Move all tensors to the device (i.e. GPU, if available)
        self._aperture_diameter = self._aperture_diameter.to(self._device)
        self._beam_combination_transfer_matrix = self._beam_combination_transfer_matrix.to(self._device)
        observatory_time_steps = observatory_time_steps.to(self._device)
        self._observatory_wavelength_bin_centers = self._observatory_wavelength_bin_centers.to(self._device)
        self._observatory_wavelength_bin_widths = self._observatory_wavelength_bin_widths.to(self._device)
        self._observatory_wavelength_bin_edges = self._observatory_wavelength_bin_edges.to(self._device)
        observatory_coordinates = observatory_coordinates.to(self._device)
        amplitude_perturbations = amplitude_perturbations.to(self._device)
        phase_perturbations = phase_perturbations.to(self._device)
        polarization_perturbations = polarization_perturbations.to(self._device)
        self._simulation_time_step_length = self._simulation_time_step_length.to(self._device)
        simulation_time_steps = simulation_time_steps.to(self._device)
        simulation_wavelength_bin_centers = simulation_wavelength_bin_centers.to(self._device)
        simulation_wavelength_bin_widths = simulation_wavelength_bin_widths.to(self._device)
        for index_source, source in enumerate(self._sources):
            self._sources[index_source].spectral_flux_density = source.spectral_flux_density.to(self._device)
            self._sources[index_source].sky_coordinates = source.sky_coordinates.to(self._device)
            self._sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
                self._device)
        self._unperturbed_instrument_throughput = self._unperturbed_instrument_throughput.to(self._device)

        # TODO: Estimate data size and start for loop, if memory is not sufficient

        # Generate the data
        data_generator = DataGenerator(
            self._aperture_diameter / 2,
            self._beam_combination_transfer_matrix,
            self._differential_output_pairs,
            self._device,
            self._grid_size,
            self._has_planet_orbital_motion,
            observatory_time_steps,
            self._observatory_wavelength_bin_centers,
            self._observatory_wavelength_bin_widths,
            self._observatory_wavelength_bin_edges,
            self._modulation_period,
            self._number_of_inputs,
            self._number_of_outputs,
            observatory_coordinates,
            amplitude_perturbations,
            phase_perturbations,
            polarization_perturbations,
            self._simulation_time_step_length,
            simulation_time_steps,
            simulation_wavelength_bin_centers,
            simulation_wavelength_bin_widths,
            self._sources,
            self._unperturbed_instrument_throughput
        )
        self._data = data_generator.run().cpu()

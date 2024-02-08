import numpy as np
from astropy import units as u
from numpy.random import normal
from tqdm.contrib.itertools import product

from sygn.core.entities.observation import Observation
from sygn.core.entities.observatory.observatory import Observatory
from sygn.core.entities.scene import Scene
from sygn.core.entities.settings import Settings
from sygn.core.util.grid import get_index_of_closest_value


class DataGenerator():
    def __init__(self,
                 settings: Settings,
                 observation: Observation,
                 observatory: Observatory,
                 scene: Scene):
        self.amplitude_perturbation_time_series = observatory.amplitude_perturbation_time_series
        self.aperture_radius = observatory.aperture_diameter / 2
        self.baseline_maximum = observation.baseline_maximum
        self.baseline_minimum = observation.baseline_minimum
        self.baseline_ratio = observation.baseline_ratio
        self.beam_combination_matrix = observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()
        self.differential_output_pairs = observatory.beam_combination_scheme.get_differential_output_pairs()
        self.measured_wavelength_bin_centers = observatory.wavelength_bin_centers
        self.measured_wavelength_bin_edges = observatory.wavelength_bin_edges
        self.measured_wavelength_bin_widths = observatory.wavelength_bin_widths
        self.measured_time_steps = np.linspace(0, observation.total_integration_time,
                                               int(observation.total_integration_time / observation.exposure_time))
        self.grid_size = settings.grid_size
        self.modulation_period = observation.modulation_period
        self.number_of_inputs = observatory.beam_combination_scheme.number_of_inputs
        self.number_of_outputs = observatory.beam_combination_scheme.number_of_outputs
        self.observatory = observatory
        self.optimized_differential_output = observation.optimized_differential_output
        self.optimized_star_separation = observation.optimized_star_separation
        self.optimized_wavelength = observation.optimized_wavelength
        self.phase_perturbation_time_series = observatory.phase_perturbation_time_series
        self.polarization_perturbation_time_series = observatory.polarization_perturbation_time_series
        self.sources = scene.get_all_sources()
        self.star = scene.star
        self.time_step_duration = settings.time_step_duration
        self.time_steps = settings.time_steps
        self.unperturbed_instrument_throughput = observatory.unperturbed_instrument_throughput
        self.wavelength_steps = settings.wavelength_steps
        self.differential_photon_counts = np.zeros((len(self.differential_output_pairs),
                                                    len(self.measured_wavelength_bin_centers),
                                                    len(self.measured_time_steps))) * u.ph
        self.photon_counts_binned = np.zeros((self.number_of_outputs,
                                              len(self.measured_wavelength_bin_centers),
                                              len(self.measured_time_steps))) * u.ph

    def _apply_shot_noise(self, mean_photon_counts):
        if mean_photon_counts > 30:
            return round(normal(mean_photon_counts, 1))
        return np.random.poisson(mean_photon_counts)

    def _calculate_complex_amplitude_base(self, index_input, index_time, wavelength, observatory_coordinates,
                                          source_sky_coordinates):
        return (self.amplitude_perturbation_time_series[index_input][index_time] * self.aperture_radius.to(u.m)
                * np.exp(1j * 2 * np.pi / wavelength * (
                        observatory_coordinates.x[index_input] * source_sky_coordinates.x.to(u.rad).value +
                        observatory_coordinates.y[index_input] * source_sky_coordinates.y.to(u.rad).value +
                        self.phase_perturbation_time_series[index_input][index_time].to(u.um))))

    def _calculate_complex_amplitude(self, time, wavelength, source_sky_coordinates):
        complex_amplitude = np.zeros((self.number_of_inputs, 2, self.grid_size, self.grid_size), dtype=complex) * u.m
        observatory_coordinates = self.observatory.array_configuration.get_collector_positions(
            time,
            self.modulation_period,
            self.baseline_ratio
        )
        index_time = int(np.where(self.time_steps == time)[0])
        polarization_angle = 0 * u.rad  # TODO: Check that we can set this to 0 without loss of generality

        for index_input in range(self.number_of_inputs):
            complex_amplitude[index_input][0] = (
                    self._calculate_complex_amplitude_base(index_input, index_time, wavelength, observatory_coordinates,
                                                           source_sky_coordinates)
                    * np.cos(polarization_angle + self.polarization_perturbation_time_series[index_input][index_time]
                             .to(u.rad)))

            complex_amplitude[index_input][1] = (
                    self._calculate_complex_amplitude_base(index_input, index_time, wavelength, observatory_coordinates,
                                                           source_sky_coordinates)
                    * np.sin(polarization_angle + self.polarization_perturbation_time_series[index_input][index_time]
                             .to(u.rad)))
        return complex_amplitude

    def _calculate_intensity_response(self, time, wavelength, source):
        complex_amplitude = (self._calculate_complex_amplitude(time, wavelength, source.sky_coordinates)
                             .reshape(self.number_of_inputs, 2, self.grid_size ** 2))
        return ((abs(np.dot(self.beam_combination_matrix, complex_amplitude[:, 0])) ** 2 +
                 abs(np.dot(self.beam_combination_matrix, complex_amplitude[:, 1])) ** 2)
                .reshape(self.number_of_outputs, self.grid_size, self.grid_size))

    def _calculate_normalization(self, source_sky_brightness_distribution):
        return len(source_sky_brightness_distribution[source_sky_brightness_distribution.value > 0]) if not len(
            source_sky_brightness_distribution[source_sky_brightness_distribution.value > 0]) == 0 else 1

    def _calculate_photon_counts(self, wavelength, source_sky_brightness_distribution,
                                 intensity_response):
        photon_counts = np.zeros(self.number_of_outputs)
        normalization = self._calculate_normalization(source_sky_brightness_distribution)
        # TODO: Note this only holds for regular wavelength steps
        wavelength_bin_width = self.wavelength_steps[1] - self.wavelength_steps[0]

        for index_ir, intensity_response in enumerate(intensity_response):
            mean_photon_counts = (
                    np.sum(intensity_response * source_sky_brightness_distribution * self.time_step_duration.to(u.s)
                           * wavelength_bin_width * self.unperturbed_instrument_throughput).value
                    / normalization)
            photon_counts[index_ir] = self._apply_shot_noise(mean_photon_counts)
        return photon_counts * u.ph

    def _get_binning_indices(self, time, wavelength):
        index_closest_wavelength_edge = get_index_of_closest_value(self.measured_wavelength_bin_edges, wavelength)
        if wavelength <= self.measured_wavelength_bin_edges[index_closest_wavelength_edge]:
            index_wavelength = index_closest_wavelength_edge - 1
        else:
            index_wavelength = index_closest_wavelength_edge
        index_closest_time_edge = get_index_of_closest_value(self.measured_time_steps, time)
        if time <= self.measured_time_steps[index_closest_time_edge]:
            index_time = index_closest_time_edge - 1
        else:
            index_time = index_closest_time_edge
        return index_wavelength, index_time

    def run(self) -> dict:
        # Set optimal baseline length
        self.observatory.set_optimal_baseline(self.star, self.optimized_differential_output, self.optimized_wavelength,
                                              self.optimized_star_separation, self.baseline_minimum,
                                              self.baseline_maximum)

        # Run animation, if applicable
        # TODO: add animation

        # Start time, wavelength and source loop
        for time, wavelength, source in product(self.time_steps, self.wavelength_steps, self.sources):

            # Calculate intensity response
            intensity_response = self._calculate_intensity_response(time, wavelength, source)

            # Calc photon counts
            photon_counts = self._calculate_photon_counts(wavelength, source.sky_brightness_distribution,
                                                          intensity_response)

            # Bin the photon counts into the measured time and wavelength intervals
            index_wavelength, index_time = self._get_binning_indices(time, wavelength)
            self.photon_counts_binned[:, index_wavelength, index_time] += photon_counts

            # Calc differential photon counts
            for index_pair, pair in enumerate(self.differential_output_pairs):
                self.differential_photon_counts[index_pair] = self.photon_counts_binned[pair[0]] - \
                                                              self.photon_counts_binned[pair[1]]

        return self.differential_photon_counts
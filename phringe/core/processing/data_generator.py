import numpy as np
from astropy import units as u
from numba import jit, complex128, float64, uint64
from numpy.random import normal, poisson
from tqdm.contrib.itertools import product

from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.util.grid import get_index_of_closest_value
from phringe.util.helpers import Coordinates


@jit(complex128[:, :](float64, uint64, uint64, float64, float64[:, :], float64[:, :], float64[:], float64[:],
                      float64[:, :], float64[:, :]), nopython=True, nogil=True, fastmath=True)
def _calculate_complex_amplitude_base(
        aperture_radius: float,
        index_input: int,
        index_time: int,
        wavelength: float,
        amplitude_perturbation_time_series: np.ndarray,
        phase_perturbation_time_series: np.ndarray,
        observatory_coordinates_x: np.ndarray,
        observatory_coordinates_y: np.ndarray,
        source_sky_coordinates_x: np.ndarray,
        source_sky_coordinates_y: np.ndarray,
) -> np.ndarray:
    """Calculate the complex amplitude element for a single polarization.

    :param index_input: The index of the input
    :param index_time: The index of the time
    :param wavelength: The wavelength
    :param observatory_coordinates: The observatory coordinates
    :param source_sky_coordinates: The source sky coordinates
    :return: The complex amplitude element
    """
    return (amplitude_perturbation_time_series[index_input][index_time] * aperture_radius
            * np.exp(1j * 2 * np.pi / wavelength * (
                    observatory_coordinates_x[index_input] * source_sky_coordinates_x +
                    observatory_coordinates_y[index_input] * source_sky_coordinates_y +
                    phase_perturbation_time_series[index_input][index_time])))


class DataGenerator():
    """Class representation of the data generator. This class is responsible for generating the synthetic photometry
     data for space-based nulling interferometers.

    :param amplitude_perturbation_time_series: The amplitude perturbation time series
    :param aperture_radius: The aperture radius
    :param baseline_maximum: The maximum baseline
    :param baseline_minimum: The minimum baseline
    :param baseline_ratio: The baseline ratio
    :param beam_combination_matrix: The beam combination matrix
    :param differential_output_pairs: The differential output pairs
    :param measured_wavelength_bin_centers: The measured wavelength bin centers
    :param measured_wavelength_bin_edges: The measured wavelength bin edges
    :param measured_wavelength_bin_widths: The measured wavelength bin widths
    :param measured_time_steps: The measured time steps
    :param grid_size: The grid size
    :param has_planet_orbital_motion: The flag indicating whether the planet has orbital motion
    :param modulation_period: The modulation period
    :param number_of_inputs: The number of inputs
    :param number_of_outputs: The number of outputs
    :param observatory: The observatory
    :param optimized_differential_output: The optimized differential output
    :param optimized_star_separation: The optimized star separation
    :param optimized_wavelength: The optimized wavelength
    :param phase_perturbation_time_series: The phase perturbation time series
    :param polarization_perturbation_time_series: The polarization perturbation time series
    :param sources: The sources
    :param star: The star
    :param time_step_duration: The time step duration
    :param time_steps: The time steps
    :param unperturbed_instrument_throughput: The unperturbed instrument throughput
    :param wavelength_steps: The wavelength steps
    :param differential_photon_counts: The differential photon counts
    :param photon_counts_binned: The photon counts binned
    """

    def __init__(self,
                 settings: Settings,
                 observation: Observation,
                 observatory: Observatory,
                 scene: Scene):
        """Constructor method.

        :param settings: The settings object
        :param observation: The observation object
        :param observatory: The observatory object
        :param scene: The scene object
        """
        self.amplitude_perturbation_time_series = observatory.amplitude_perturbation_time_series
        self.aperture_radius = observatory.aperture_diameter.to(u.m).value / 2
        self.baseline_maximum = observation.baseline_maximum.to(u.m).value
        self.baseline_minimum = observation.baseline_minimum.to(u.m).value
        self.baseline_ratio = observation.baseline_ratio
        self.beam_combination_matrix = observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()
        self.differential_output_pairs = observatory.beam_combination_scheme.get_differential_output_pairs()
        self.measured_wavelength_bin_centers = observatory.wavelength_bin_centers.to(u.m).value
        self.measured_wavelength_bin_edges = observatory.wavelength_bin_edges.to(u.m).value
        self.measured_wavelength_bin_widths = observatory.wavelength_bin_widths.to(u.m).value
        self.measured_time_steps = np.linspace(0, observation.total_integration_time,
                                               int(observation.total_integration_time / observation.exposure_time)).to(
            u.s).value
        self.grid_size = settings.grid_size
        self.has_planet_orbital_motion = settings.has_planet_orbital_motion
        self.modulation_period = observation.modulation_period.to(u.s).value
        self.number_of_inputs = observatory.beam_combination_scheme.number_of_inputs
        self.number_of_outputs = observatory.beam_combination_scheme.number_of_outputs
        self.observatory = observatory
        self.optimized_differential_output = observation.optimized_differential_output
        self.optimized_star_separation = observation.optimized_star_separation
        self.optimized_wavelength = observation.optimized_wavelength.to(u.m).value
        self.phase_perturbation_time_series = observatory.phase_perturbation_time_series.to(u.m).value
        self.polarization_perturbation_time_series = observatory.polarization_perturbation_time_series.to(u.rad).value
        self.sources = scene.get_all_sources(
            settings.has_stellar_leakage,
            settings.has_local_zodi_leakage,
            settings.has_exozodi_leakage
        )
        self.star = scene.star
        self.time_step_duration = settings.time_step_duration.to(u.s).value
        self.time_steps = settings.time_steps.to(u.s).value
        self.unperturbed_instrument_throughput = observatory.unperturbed_instrument_throughput
        self.wavelength_steps = settings.wavelength_steps.to(u.m).value
        self.differential_photon_counts = np.zeros((len(self.differential_output_pairs),
                                                    len(self.measured_wavelength_bin_centers),
                                                    len(self.measured_time_steps)))
        self.photon_counts_binned = np.zeros((self.number_of_outputs,
                                              len(self.measured_wavelength_bin_centers),
                                              len(self.measured_time_steps)))
        self._remove_units_from_source_sky_coordinates()
        self._remove_units_from_source_sky_brightness_distribution()
        self._remove_units_from_collector_coordinates()

    def _apply_shot_noise(self, mean_photon_counts) -> int:
        """Apply shot noise to the mean photon counts.

        :param mean_photon_counts: The mean photon counts
        :return: The value corresponding to the expected shot noise
        """

        try:
            photon_counts = poisson(mean_photon_counts, 1)
        except ValueError:
            photon_counts = round(normal(mean_photon_counts, 1))
        return photon_counts

    def _calculate_complex_amplitude(self, time, wavelength, source) -> np.ndarray:
        """Calculate the complex amplitude.

        :param time: The time
        :param wavelength: The wavelength
        :param source: The source
        :return: The complex amplitude
        """
        index_time = int(np.where(self.time_steps == time)[0])
        index_wavelength = int(np.where(self.wavelength_steps == wavelength)[0])
        complex_amplitude = np.zeros((self.number_of_inputs, 2, self.grid_size, self.grid_size), dtype=complex)
        observatory_coordinates = self.observatory.array_configuration.collector_coordinates[index_time]
        polarization_angle = 0  # TODO: Check that we can set this to 0 without loss of generality

        if self.has_planet_orbital_motion and isinstance(source, Planet):
            source_sky_coordinates = source.sky_coordinates[index_time]
        elif isinstance(source, LocalZodi) or isinstance(source, Exozodi):
            source_sky_coordinates = source.sky_coordinates[index_wavelength]
        else:
            source_sky_coordinates = source.sky_coordinates

        for index_input in range(self.number_of_inputs):
            base_complex_amplitude = _calculate_complex_amplitude_base(
                self.aperture_radius,
                index_input,
                index_time,
                wavelength,
                self.amplitude_perturbation_time_series,
                self.phase_perturbation_time_series,
                observatory_coordinates.x,
                observatory_coordinates.y,
                source_sky_coordinates.x,
                source_sky_coordinates.y
            )
            complex_amplitude[index_input][0] = (base_complex_amplitude * np.cos(
                polarization_angle + self.polarization_perturbation_time_series[index_input][index_time]))

            complex_amplitude[index_input][1] = (base_complex_amplitude * np.sin(
                polarization_angle + self.polarization_perturbation_time_series[index_input][index_time]))
        return complex_amplitude

    def _calculate_intensity_response(self, time, wavelength, source) -> np.ndarray:
        """Calculate the intensity response.

        :param time: The time
        :param wavelength: The wavelength
        :param source: The source
        :return: The intensity response
        """
        complex_amplitude = (self._calculate_complex_amplitude(time, wavelength, source)
                             .reshape(self.number_of_inputs, 2, self.grid_size ** 2))
        return ((abs(np.dot(self.beam_combination_matrix, complex_amplitude[:, 0])) ** 2 +
                 abs(np.dot(self.beam_combination_matrix, complex_amplitude[:, 1])) ** 2)
                .reshape(self.number_of_outputs, self.grid_size, self.grid_size))

    def _calculate_normalization(self, source_sky_brightness_distribution, index_wavelength: int) -> int:
        """Calculate the normalization.

        :param source_sky_brightness_distribution: The source sky brightness distribution
        :return: The normalization
        """
        source_sky_brightness_distribution = source_sky_brightness_distribution[index_wavelength]
        return len(source_sky_brightness_distribution[source_sky_brightness_distribution > 0]) if not len(
            source_sky_brightness_distribution[source_sky_brightness_distribution > 0]) == 0 else 1

    def _calculate_photon_counts(
            self,
            time,
            wavelength,
            source,
            intensity_response
    ) -> np.ndarray:
        """Calculate the photon counts.

        :param time: The time
        :param wavelength: The wavelength
        :param source: The source
        :param intensity_response: The intensity response
        :return: The photon counts
        """
        if self.has_planet_orbital_motion and isinstance(source, Planet):
            index_time = int(np.where(self.time_steps == time)[0])
            source_sky_brightness_distribution = source.sky_brightness_distribution[index_time]
        else:
            source_sky_brightness_distribution = source.sky_brightness_distribution

        photon_counts = np.zeros(self.number_of_outputs)
        index_wavelength = int(np.where(self.wavelength_steps == wavelength)[0])
        normalization = self._calculate_normalization(source_sky_brightness_distribution, index_wavelength)
        # TODO: Note this only holds for regular wavelength steps
        wavelength_bin_width = self.wavelength_steps[1] - self.wavelength_steps[0]

        for index_ir, intensity_response in enumerate(intensity_response):
            mean_photon_counts = (
                    np.sum(intensity_response
                           * source_sky_brightness_distribution[index_wavelength]
                           * self.time_step_duration
                           * wavelength_bin_width
                           * self.unperturbed_instrument_throughput)
                    / normalization)
            photon_counts[index_ir] = self._apply_shot_noise(mean_photon_counts)
        return photon_counts

    def _get_binning_indices(self, time, wavelength) -> tuple:
        """Get the binning indices.

        :param time: The time
        :param wavelength: The wavelength
        :return: The binning indices
        """
        index_closest_wavelength_edge = get_index_of_closest_value(self.measured_wavelength_bin_edges, wavelength)
        if index_closest_wavelength_edge == 0:
            index_wavelength_bin = 0
        elif wavelength <= self.measured_wavelength_bin_edges[index_closest_wavelength_edge]:
            index_wavelength_bin = index_closest_wavelength_edge - 1
        else:
            index_wavelength_bin = index_closest_wavelength_edge
        index_closest_time_edge = get_index_of_closest_value(self.measured_time_steps, time)
        if index_closest_time_edge == 0:
            index_time = 0
        elif time <= self.measured_time_steps[index_closest_time_edge]:
            index_time = index_closest_time_edge - 1
        else:
            index_time = index_closest_time_edge
        return index_wavelength_bin, index_time

    def _remove_units_from_source_sky_coordinates(self):
        for index_source, source in enumerate(self.sources):
            if self.has_planet_orbital_motion and isinstance(source, Planet):
                for index_time, time in enumerate(self.time_steps):
                    self.sources[index_source].sky_coordinates[index_time] = Coordinates(
                        source.sky_coordinates[index_time].x.to(u.rad).value,
                        source.sky_coordinates[index_time].y.to(u.rad).value
                    )
            elif isinstance(source, LocalZodi) or isinstance(source, Exozodi):
                for index_wavelength, wavelength in enumerate(self.wavelength_steps):
                    self.sources[index_source].sky_coordinates[index_wavelength] = Coordinates(
                        source.sky_coordinates[index_wavelength].x.to(u.rad).value,
                        source.sky_coordinates[index_wavelength].y.to(u.rad).value
                    )
            else:
                self.sources[index_source].sky_coordinates = Coordinates(
                    source.sky_coordinates.x.to(u.rad).value,
                    source.sky_coordinates.y.to(u.rad).value
                )

    def _remove_units_from_source_sky_brightness_distribution(self):
        for index_source, source in enumerate(self.sources):
            self.sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
                u.ph / (u.m ** 3 * u.s)).value

    def _remove_units_from_collector_coordinates(self):
        for index_time, time in enumerate(self.time_steps):
            self.observatory.array_configuration.collector_coordinates[index_time] = Coordinates(
                self.observatory.array_configuration.collector_coordinates[index_time].x.to(u.m).value,
                self.observatory.array_configuration.collector_coordinates[index_time].y.to(u.m).value
            )

    def run(self) -> np.ndarray:
        """Run the data generator.
        """
        # Run animation, if applicable
        # TODO: add animation

        # Start time, wavelength and source loop
        for time, wavelength, source in product(self.time_steps, self.wavelength_steps, self.sources):

            # Calculate intensity response
            intensity_response = self._calculate_intensity_response(time, wavelength, source)

            # Calculate photon counts
            photon_counts = self._calculate_photon_counts(
                time,
                wavelength,
                source,
                intensity_response
            )

            # Bin the photon counts into the measured time and wavelength intervals
            index_wavelength, index_time = self._get_binning_indices(time, wavelength)
            self.photon_counts_binned[:, index_wavelength, index_time] += photon_counts

            # Calculate differential photon counts
            for index_pair, pair in enumerate(self.differential_output_pairs):
                self.differential_photon_counts[index_pair] = self.photon_counts_binned[pair[0]] - \
                                                              self.photon_counts_binned[pair[1]]

        return self.differential_photon_counts * u.ph

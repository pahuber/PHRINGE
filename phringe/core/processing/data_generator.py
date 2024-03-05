import time

import numpy as np
import torch
from astropy import units as u

from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.core.processing.processing import calculate_photon_counts_gpu
from phringe.util.grid import get_index_of_closest_value_numpy


# @jit(complex128[:, :, :, :, :](float64, float64, float64[:, :], float64[:, :], float64[:, :], float64[:, :],
#                                float64[:, :], float64[:, :], float64[:], uint64), nopython=True, nogil=True,
#      fastmath=True)
# @jit(complex128[:, :, :, :, :](float64, float64, float64[:, :], float64[:, :], float64[:, :], float64[:, :],
#                                float64[:, :], float64[:, :], float64[:], uint64), nopython=True, nogil=True,
#      fastmath=True)
# def _calculate_complex_amplitude_base(
#         aperture_radius: float,
#         unperturbed_instrument_throughput: float,
#         amplitude_perturbation_time_series: np.ndarray,
#         phase_perturbation_time_series: np.ndarray,
#         observatory_coordinates_x: np.ndarray,
#         observatory_coordinates_y: np.ndarray,
#         source_sky_coordinates_x: np.ndarray,
#         source_sky_coordinates_y: np.ndarray,
#         wavelength_steps: np.ndarray,
#         grid_size: int
# ) -> np.ndarray:
#     """Calculate the complex amplitude element for a single polarization.
#
#     :param unperturbed_instrument_throughput: The unperturbed instrument throughput
#     :param amplitude_perturbation_time_series: The amplitude perturbation time series
#     :param phase_perturbation_time_series: The phase perturbation time series
#     :param observatory_coordinates_x: The observatory x coordinates
#     :param observatory_coordinates_y: The observatory y coordinates
#     :param source_sky_coordinates_x: The source sky x coordinates
#     :param source_sky_coordinates_y: The source sky y coordinates
#     :return: The complex amplitude element
#     """
#     const = aperture_radius * torch.sqrt(unperturbed_instrument_throughput)
#     exp_const = 2j * torch.pi
#
#     obs_x_source_x = (
#             observatory_coordinates_x[..., None, None] *
#             source_sky_coordinates_x[None, None, ...])
#
#     obs_y_source_y = (
#             observatory_coordinates_y[..., None, None] *
#             source_sky_coordinates_y[None, None, ...])
#
#     phase_pert = phase_perturbation_time_series[..., None, None]
#
#     sum = obs_x_source_x + obs_y_source_y + phase_pert
#
#     exp = (exp_const * (1 / wavelength_steps)[..., None, None, None, None] *
#            (obs_x_source_x + obs_y_source_y + phase_pert)[None, ...])
#
#     a = const * amplitude_perturbation_time_series[None, ..., None, None] * torch.exp(exp)
#
#     return a


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
    :param generate_separate: The flag indicating whether to enable photon statistics by generating separate data sets for all
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
                 scene: Scene,
                 generate_separate: bool):
        """Constructor method.

        :param settings: The settings object
        :param observation: The observation object
        :param observatory: The observatory object
        :param scene: The scene object
        :param generate_separate: Whether to separate data sets for all sources
        """
        self.aperture_radius = observatory.aperture_diameter.to(u.m).value / 2
        self.baseline_maximum = observation.baseline_maximum.to(u.m).value
        self.baseline_minimum = observation.baseline_minimum.to(u.m).value
        self.baseline_ratio = observation.baseline_ratio
        self.differential_output_pairs = observatory.beam_combination_scheme.get_differential_output_pairs()
        self.generate_separate = generate_separate
        self.instrument_wavelength_bin_centers = observatory.wavelength_bin_centers.to(u.m).value
        self.instrument_wavelength_bin_widths = observatory.wavelength_bin_widths.to(u.m).value
        self.grid_size = settings.grid_size
        self.has_planet_orbital_motion = settings.has_planet_orbital_motion
        self.modulation_period = observation.modulation_period.to(u.s).value
        self.number_of_inputs = observatory.beam_combination_scheme.number_of_inputs
        self.number_of_outputs = observatory.beam_combination_scheme.number_of_outputs
        self.observatory = observatory
        self.optimized_differential_output = observation.optimized_differential_output
        self.optimized_star_separation = observation.optimized_star_separation
        self.optimized_wavelength = observation.optimized_wavelength.to(u.m).value
        self.sources = scene.get_all_sources(
            settings.has_stellar_leakage,
            settings.has_local_zodi_leakage,
            settings.has_exozodi_leakage
        )
        self.star = scene.star
        self.simulation_time_step_duration = settings.simulation_time_step_duration.to(u.s).value
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        # torch.set_num_threads(10)

        # GPU stuff starts here
        self.unperturbed_instrument_throughput = observatory.unperturbed_instrument_throughput
        self.instrument_wavelength_bin_edges = observatory.wavelength_bin_edges.to(u.m).value
        self.instrument_time_steps = np.linspace(
            0,
            observation.total_integration_time,
            int(observation.total_integration_time / observation.detector_integration_time)
        ).to(u.s).value
        self.amplitude_perturbation_time_series = observatory.amplitude_perturbation_time_series
        self.beam_combination_matrix = observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()
        self.phase_perturbation_time_series = observatory.phase_perturbation_time_series.to(u.m).value
        self.polarization_perturbation_time_series = observatory.polarization_perturbation_time_series.to(u.rad).value
        self.simulation_time_steps = settings.simulation_time_steps.to(u.s).value
        self.simulation_wavelength_steps = settings.simulation_wavelength_steps.to(u.m).value
        self.simulation_wavelength_bin_widths = settings.simulation_wavelength_bin_widths.to(u.m).value
        self.binned_photon_counts = self._initialize_binned_photon_counts()
        self.differential_photon_counts = self._initialize_differential_photon_counts()
        self._remove_units_from_source_sky_coordinates()
        self._remove_units_from_source_sky_brightness_distribution()
        self._remove_units_from_collector_coordinates()

    def _get_binning_indices(self, time, wavelength) -> tuple:
        """Get the binning indices.

        :param time: The time
        :param wavelength: The wavelength
        :return: The binning indices
        """
        index_closest_wavelength_edge = get_index_of_closest_value_numpy(
            self.instrument_wavelength_bin_edges,
            wavelength
        )
        if index_closest_wavelength_edge == 0:
            index_wavelength_bin = 0
        elif wavelength <= self.instrument_wavelength_bin_edges[index_closest_wavelength_edge]:
            index_wavelength_bin = index_closest_wavelength_edge - 1
        else:
            index_wavelength_bin = index_closest_wavelength_edge

        index_closest_time_edge = get_index_of_closest_value_numpy(self.instrument_time_steps, time)
        if index_closest_time_edge == 0:
            index_time = 0
        elif time <= self.instrument_time_steps[index_closest_time_edge]:
            index_time = index_closest_time_edge - 1
        else:
            index_time = index_closest_time_edge
        return index_wavelength_bin, index_time

    def _initialize_binned_photon_counts(self):
        binned_photon_counts = torch.zeros(
            (self.number_of_outputs, len(self.instrument_wavelength_bin_centers), len(self.instrument_time_steps)),
            dtype=torch.float32,
            device='cpu'
        )
        binned_photon_counts = {source.name: binned_photon_counts.detach().clone() for source in
                                self.sources} if self.generate_separate else binned_photon_counts
        return binned_photon_counts

    def _initialize_differential_photon_counts(self):
        differential_photon_counts = torch.zeros(
            (
                len(self.differential_output_pairs),
                len(self.instrument_wavelength_bin_centers),
                len(self.instrument_time_steps)
            ),
            dtype=torch.float32,
            device='cpu'
        )
        differential_photon_counts = {source.name: differential_photon_counts.detach().clone() for source in
                                      self.sources} if self.generate_separate else differential_photon_counts
        return differential_photon_counts

    def _remove_units_from_source_sky_coordinates(self):
        for index_source, source in enumerate(self.sources):
            # if self.has_planet_orbital_motion and isinstance(source, Planet):
            #     for index_time, time in enumerate(self.simulation_time_steps):
            #         self.sources[index_source].sky_coordinates[index_time] = Coordinates(
            #             source.sky_coordinates[index_time].x.to(u.rad).value,
            #             source.sky_coordinates[index_time].y.to(u.rad).value
            #         )
            # elif isinstance(source, LocalZodi) or isinstance(source, Exozodi):
            #     for index_wavelength, wavelength in enumerate(self.simulation_wavelength_steps):
            #         self.sources[index_source].sky_coordinates[index_wavelength] = Coordinates(
            #             source.sky_coordinates[index_wavelength].x.to(u.rad).value,
            #             source.sky_coordinates[index_wavelength].y.to(u.rad).value
            #         )
            # else:
            self.sources[index_source].sky_coordinates = self.sources[index_source].sky_coordinates.to(u.rad).value
            # self.sources[index_source].sky_coordinates = Coordinates(
            #     source.sky_coordinates.x.to(u.rad).value,
            #     source.sky_coordinates.y.to(u.rad).value
            # )

    def _remove_units_from_source_sky_brightness_distribution(self):
        for index_source, source in enumerate(self.sources):
            self.sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
                u.ph / (u.m ** 3 * u.s)).value

    def _remove_units_from_collector_coordinates(self):
        self.observatory.array_configuration.collector_coordinates = self.observatory.array_configuration.collector_coordinates.to(
            u.m).value
        # for index_time, time in enumerate(self.simulation_time_steps):
        #     self.observatory.array_configuration.collector_coordinates[index_time] = Coordinates(
        #         self.observatory.array_configuration.collector_coordinates[index_time].x.to(u.m).value,
        #         self.observatory.array_configuration.collector_coordinates[index_time].y.to(u.m).value
        #     )

    def run(self) -> np.ndarray:
        """Run the data generator.
        """
        # Run animation, if applicable
        # TODO: add animation

        self.unperturbed_instrument_throughput = torch.asarray(
            self.unperturbed_instrument_throughput,
            dtype=torch.float32
        ).to(self.device)
        self.aperture_radius = torch.tensor(self.aperture_radius, dtype=torch.float32).to(self.device)
        self.amplitude_perturbation_time_series = torch.asarray(
            self.amplitude_perturbation_time_series,
            dtype=torch.float32
        ).to(self.device)
        self.phase_perturbation_time_series = torch.asarray(
            self.phase_perturbation_time_series,
            dtype=torch.float32
        ).to(self.device)
        self.polarization_perturbation_time_series = torch.asarray(
            self.polarization_perturbation_time_series,
            dtype=torch.float32
        ).to(self.device)
        self.observatory.array_configuration.collector_coordinates = torch.asarray(
            self.observatory.array_configuration.collector_coordinates,
            dtype=torch.float32
        ).to(self.device)
        self.simulation_wavelength_bin_widths = torch.asarray(
            self.simulation_wavelength_bin_widths,
            dtype=torch.float32
        ).to(self.device)
        self.simulation_time_step_duration = torch.tensor(
            self.simulation_time_step_duration,
            dtype=torch.float32
        ).to(self.device)
        self.beam_combination_matrix = torch.asarray(
            self.beam_combination_matrix,
            dtype=torch.complex64
        ).to(self.device)

        # Start time, wavelength and source loop
        for source in self.sources:

            #
            # photon_counts = torch.zeros(
            #     (self.number_of_outputs, len(self.simulation_wavelength_steps), len(self.simulation_time_steps)),
            #     device=self.device
            # )
            # source.sky_coordinates = torch.asarray(source.sky_coordinates).to(self.device)
            # source.sky_brightness_distribution = torch.asarray(source.sky_brightness_distribution).to(self.device)

            # Calculate intensity response
            # intensity_response = self._calculate_intensity_response(source)
            #
            # # Calculate photon counts
            # photon_counts = self._calculate_photon_counts(source, intensity_response)
            # intensity_response.to('cpu')

            # if self.has_planet_orbital_motion and isinstance(source, Planet):
            #     index_time = int(np.where(self.simulation_time_steps == time)[0])
            #     source_sky_brightness_distribution = source.sky_brightness_distribution[index_time]
            # else:
            source_sky_brightness_distribution = source.sky_brightness_distribution
            # if self.has_planet_orbital_motion and isinstance(source, Planet):
            #     source_sky_coordinates = source.sky_coordinates
            # elif isinstance(source, LocalZodi) or isinstance(source, Exozodi):
            #     source_sky_coordinates = source.sky_coordinates
            # else:
            source_sky_coordinates = source.sky_coordinates

            # t0 = time.time_ns()
            source_sky_coordinates = torch.asarray(
                source_sky_coordinates,
                dtype=torch.float32
            ).to(self.device)
            source_sky_brightness_distribution = torch.asarray(
                source_sky_brightness_distribution,
                dtype=torch.float32
            ).to(self.device)
            self.simulation_wavelength_steps = torch.asarray(
                self.simulation_wavelength_steps,
                dtype=torch.float32
            ).to(self.device)

            # self.simulation_time_steps = torch.asarray(
            #     self.simulation_time_steps,
            #     dtype=torch.float32
            # ).to(self.device)
            # self.instrument_time_steps = torch.asarray(
            #     self.instrument_time_steps,
            #     dtype=torch.float32
            # ).to(self.device)
            # self.instrument_wavelength_bin_edges = torch.asarray(
            #     self.instrument_wavelength_bin_edges,
            #     dtype=torch.float32
            # ).to(self.device)
            # self.differential_output_pairs = torch.asarray(
            #     self.differential_output_pairs,
            #     dtype=torch.int32
            # ).to(self.device)

            # t1 = time.time_ns()
            # print(f'Elapsed time source: {(t1 - t0) / 1e9} s')
            t0 = time.time_ns()

            photon_counts = calculate_photon_counts_gpu(
                self.device,
                self.aperture_radius,
                self.unperturbed_instrument_throughput,
                self.amplitude_perturbation_time_series,
                self.phase_perturbation_time_series,
                self.polarization_perturbation_time_series,
                self.observatory.array_configuration.collector_coordinates[0],
                self.observatory.array_configuration.collector_coordinates[1],
                source_sky_coordinates[0],
                source_sky_coordinates[1],
                source_sky_brightness_distribution,
                self.simulation_wavelength_steps,
                self.simulation_wavelength_bin_widths,
                self.simulation_time_step_duration,
                self.beam_combination_matrix,
                torch.empty(len(source_sky_brightness_distribution), device=self.device)
            )

            # self.unperturbed_instrument_throughput = self.unperturbed_instrument_throughput.to('cpu')
            t1 = time.time_ns()

            print(f'Elapsed time source: {(t1 - t0) / 1e9} s')
            t0 = time.time_ns()

            # self.simulation_time_steps = self.simulation_time_steps.to('cpu')
            self.simulation_wavelength_steps = self.simulation_wavelength_steps.to('cpu').numpy()
            photon_counts = photon_counts.to('cpu')
            # self.instrument_wavelength_bin_edges = self.instrument_wavelength_bin_edges.to('cpu')

            # Bin the photon counts into the instrument time and wavelength intervals
            for index_time2, time2 in enumerate(self.simulation_time_steps):
                for index_wavelength2, wavelength in enumerate(self.simulation_wavelength_steps):

                    index_wavelength, index_time = self._get_binning_indices(time2, wavelength)
                    # print(f'{index_time2, index_wavelength2} -> {index_time, index_wavelength}')
                    if self.generate_separate:
                        self.binned_photon_counts[source.name][:, index_wavelength, index_time] += photon_counts[
                                                                                                   :,
                                                                                                   index_wavelength2,
                                                                                                   index_time2]
                    else:
                        self.binned_photon_counts[:, index_wavelength, index_time] += photon_counts[:,
                                                                                      index_wavelength2,
                                                                                      index_time2]

            t1 = time.time_ns()
            print(f'Elapsed time binning: {(t1 - t0) / 1e9} s')

            t0 = time.time_ns()

            # Calculate differential photon counts
            for index_pair, pair in enumerate(self.differential_output_pairs):
                if self.generate_separate:
                    for source in self.sources:
                        self.differential_photon_counts[source.name][index_pair] = \
                            (
                                    self.binned_photon_counts[source.name][pair[0]] - \
                                    self.binned_photon_counts[source.name][pair[1]]
                            )
                else:
                    self.differential_photon_counts[index_pair] = self.binned_photon_counts[pair[0]] - \
                                                                  self.binned_photon_counts[pair[1]]

            t1 = time.time_ns()

            print(f'Elapsed time ph: {(t1 - t0) / 1e9} s')

        return self.differential_photon_counts

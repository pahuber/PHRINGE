import time

import numpy as np
import torch
from skimage.measure import block_reduce
from torch import Tensor

from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.core.processing.processing import calculate_photon_counts_gpu
from phringe.util.grid import get_index_of_closest_value_numpy


# from astropy.nddata import block_reduce


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
        self.aperture_radius = observatory.aperture_diameter / 2
        self.baseline_maximum = observation.baseline_maximum
        self.baseline_minimum = observation.baseline_minimum
        self.baseline_ratio = observation.baseline_ratio
        self.differential_output_pairs = observatory.beam_combination_scheme.get_differential_output_pairs()
        self.generate_separate = generate_separate
        self.instrument_wavelength_bin_centers = observatory.wavelength_bin_centers
        self.instrument_wavelength_bin_widths = observatory.wavelength_bin_widths
        self.grid_size = settings.grid_size
        self.has_planet_orbital_motion = settings.has_planet_orbital_motion
        self.modulation_period = observation.modulation_period
        self.number_of_inputs = observatory.beam_combination_scheme.number_of_inputs
        self.number_of_outputs = observatory.beam_combination_scheme.number_of_outputs
        self.observatory = observatory
        self.optimized_differential_output = observation.optimized_differential_output
        self.optimized_star_separation = observation.optimized_star_separation
        self.optimized_wavelength = observation.optimized_wavelength
        self.sources = scene.get_all_sources(
            settings.has_stellar_leakage,
            settings.has_local_zodi_leakage,
            settings.has_exozodi_leakage
        )
        self.star = scene.star
        self.simulation_time_step_duration = settings.simulation_time_step_duration
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        # torch.set_num_threads(1)

        # GPU stuff starts here
        self.unperturbed_instrument_throughput = observatory.unperturbed_instrument_throughput
        self.instrument_wavelength_bin_edges = observatory.wavelength_bin_edges
        self.instrument_time_steps = torch.linspace(
            0,
            observation.total_integration_time,
            int(observation.total_integration_time / observation.detector_integration_time)
        )
        self.amplitude_perturbation_time_series = observatory.amplitude_perturbation_time_series
        self.beam_combination_matrix = observatory.beam_combination_scheme.get_beam_combination_transfer_matrix()
        self.phase_perturbation_time_series = observatory.phase_perturbation_time_series
        self.polarization_perturbation_time_series = observatory.polarization_perturbation_time_series
        self.simulation_time_steps = settings.simulation_time_steps
        self.simulation_wavelength_steps = settings.simulation_wavelength_steps
        self.simulation_wavelength_bin_widths = settings.simulation_wavelength_bin_widths
        self.binned_photon_counts = self._initialize_binned_photon_counts()
        self.differential_photon_counts = self._initialize_differential_photon_counts()
        # self._remove_units_from_source_sky_coordinates()
        # self._remove_units_from_source_sky_brightness_distribution()
        # self._remove_units_from_collector_coordinates()

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

    # def _remove_units_from_source_sky_coordinates(self):
    #     for index_source, source in enumerate(self.sources):
    #         # if self.has_planet_orbital_motion and isinstance(source, Planet):
    #         #     for index_time, time in enumerate(self.simulation_time_steps):
    #         #         self.sources[index_source].sky_coordinates[index_time] = Coordinates(
    #         #             source.sky_coordinates[index_time].x.to(u.rad).value,
    #         #             source.sky_coordinates[index_time].y.to(u.rad).value
    #         #         )
    #         # elif isinstance(source, LocalZodi) or isinstance(source, Exozodi):
    #         #     for index_wavelength, wavelength in enumerate(self.simulation_wavelength_steps):
    #         #         self.sources[index_source].sky_coordinates[index_wavelength] = Coordinates(
    #         #             source.sky_coordinates[index_wavelength].x.to(u.rad).value,
    #         #             source.sky_coordinates[index_wavelength].y.to(u.rad).value
    #         #         )
    #         # else:
    #         self.sources[index_source].sky_coordinates = self.sources[index_source].sky_coordinates.to(u.rad).value
    #         # self.sources[index_source].sky_coordinates = Coordinates(
    #         #     source.sky_coordinates.x.to(u.rad).value,
    #         #     source.sky_coordinates.y.to(u.rad).value
    #         # )

    # def _remove_units_from_source_sky_brightness_distribution(self):
    #     for index_source, source in enumerate(self.sources):
    #         self.sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
    #             u.ph / (u.m ** 3 * u.s)).value

    # def _remove_units_from_collector_coordinates(self):
    #     self.observatory.array_configuration.collector_coordinates = self.observatory.array_configuration.collector_coordinates.to(
    #         u.m).value
    #     # for index_time, time in enumerate(self.simulation_time_steps):
    #     #     self.observatory.array_configuration.collector_coordinates[index_time] = Coordinates(
    #     #         self.observatory.array_configuration.collector_coordinates[index_time].x.to(u.m).value,
    #     #         self.observatory.array_configuration.collector_coordinates[index_time].y.to(u.m).value
    #     #     )

    def run(self) -> np.ndarray:
        """Run the data generator.
        """
        # Run animation, if applicable
        # TODO: add animation

        # self.unperturbed_instrument_throughput = torch.asarray(
        #     self.unperturbed_instrument_throughput,
        #     dtype=torch.float32
        # ).to(self.device)
        self.unperturbed_instrument_throughput = self.unperturbed_instrument_throughput.to(self.device)
        # self.aperture_radius = torch.tensor(self.aperture_radius, dtype=torch.float32).to(self.device)
        self.aperture_radius = self.aperture_radius.to(self.device)
        self.amplitude_perturbation_time_series = self.amplitude_perturbation_time_series.to(self.device)
        self.phase_perturbation_time_series = self.phase_perturbation_time_series.to(self.device)
        self.polarization_perturbation_time_series = self.polarization_perturbation_time_series.to(self.device)
        self.observatory.array_configuration.collector_coordinates = self.observatory.array_configuration.collector_coordinates.to(
            self.device)
        self.simulation_wavelength_bin_widths = self.simulation_wavelength_bin_widths.to(self.device)
        self.simulation_time_step_duration = self.simulation_time_step_duration.to(self.device)
        self.beam_combination_matrix = self.beam_combination_matrix.to(self.device)

        total_photon_counts = torch.zeros(
            (self.number_of_outputs, len(self.simulation_wavelength_steps), len(self.simulation_time_steps)),
            device='cpu'
        )

        # Start time, wavelength and source loop
        for source in self.sources:
            source_sky_brightness_distribution = source.sky_brightness_distribution
            source_sky_coordinates = source.sky_coordinates

            # t0 = time.time_ns()
            source_sky_coordinates = source_sky_coordinates.to(self.device)
            source_sky_brightness_distribution = source_sky_brightness_distribution.to(self.device)
            self.simulation_wavelength_steps = self.simulation_wavelength_steps.to(self.device)

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

            # self.simulation_wavelength_steps = self.simulation_wavelength_steps.to('cpu').numpy()

            photon_counts = photon_counts.to('cpu').numpy()
            total_photon_counts += photon_counts

        # bin and calc diff

        total_photon_counts = torch.tensor(total_photon_counts, dtype=torch.float32).to(self.device)
        self.simulation_wavelength_steps = torch.tensor(self.simulation_wavelength_steps, dtype=torch.float32).to(
            self.device)
        self.instrument_wavelength_bin_edges = torch.tensor(self.instrument_wavelength_bin_edges,
                                                            dtype=torch.float32).to(
            self.device)

        print(torch.sum(total_photon_counts))

        # bin time
        self.binned_photon_counts = torch.asarray(block_reduce(total_photon_counts.cpu().numpy(),
                                                               (1, 1,
                                                                len(self.simulation_time_steps) // len(
                                                                    self.instrument_time_steps)),
                                                               np.sum)).to(self.device)

        print(torch.sum(self.binned_photon_counts))
        t0 = time.time_ns()

        # plt.imshow(self.binned_photon_counts.cpu()[2] - self.binned_photon_counts[3].cpu())
        # plt.colorbar()
        # plt.show()

        # bin wl
        self.binned_photon_counts_wl = torch.zeros(
            (self.number_of_outputs, len(self.instrument_wavelength_bin_centers), self.binned_photon_counts.shape[2]),
            dtype=torch.float32,
            device=self.device
        )

        for index_wl, wl in enumerate(self.simulation_wavelength_steps):
            index_closest_wavelength_edge = torch.abs(self.instrument_wavelength_bin_edges - wl).argmin()

            if index_closest_wavelength_edge == 0:
                index_wavelength_bin = 0
            elif wl <= self.instrument_wavelength_bin_edges[index_closest_wavelength_edge]:
                index_wavelength_bin = index_closest_wavelength_edge - 1
            else:
                index_wavelength_bin = index_closest_wavelength_edge

            self.binned_photon_counts_wl[:, index_wavelength_bin, :] += self.binned_photon_counts[:, index_wl, :]

        print(torch.sum(self.binned_photon_counts_wl))

        print(total_photon_counts.shape)
        t1 = time.time_ns()
        print(f'Elapsed time binned: {(t1 - t0) / 1e9} s')

        self.differential_photon_counts = torch.zeros(

            self.binned_photon_counts_wl.shape,

            dtype=torch.float32,
            device='cpu'
        )

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
                self.differential_photon_counts[index_pair] = Tensor(self.binned_photon_counts_wl[pair[0]] - \
                                                                     self.binned_photon_counts_wl[pair[1]])

        t1 = time.time_ns()

        return self.differential_photon_counts

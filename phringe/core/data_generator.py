import numpy as np
import torch
from skimage.measure import block_reduce
from torch import Tensor

from phringe.core.data_generator_helpers import calculate_photon_counts
from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.util.grid import get_index_of_closest_value_numpy


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

    def __init__(
            self,
            aperture_radius: float,
            beam_combination_matrix: Tensor,
            differential_output_pairs: list[tuple[int, int]],
            device: str,
            grid_size: int,
            has_planet_orbital_motion: bool,
            observatory_time_steps: Tensor,
            observatory_wavelength_bin_centers: Tensor,
            observatory_wavelength_bin_widths: Tensor,
            observatory_wavelength_bin_edges: Tensor,
            modulation_period: float,
            number_of_inputs: int,
            number_of_outputs: int,
            observatory_coordinates: Tensor,
            amplitude_perturbations: Tensor,
            phase_perturbations: Tensor,
            polarization_perturbations: Tensor,
            simulation_time_step_length: float,
            simulation_time_steps: Tensor,
            simulation_wavelength_bin_centers: Tensor,
            simulation_wavelength_bin_widths: Tensor,
            sources: list[BasePhotonSource],
            unperturbed_instrument_throughput: Tensor,
    ):
        """Constructor method.

        :param settings: The settings object
        :param observation: The observation object
        :param observatory: The observatory object
        :param scene: The scene object
        :param generate_separate: Whether to separate data sets for all sources
        """
        self.aperture_radius = aperture_radius
        self.beam_combination_matrix = beam_combination_matrix
        self.device = device
        self.instrument_wavelength_bin_centers = observatory_wavelength_bin_centers
        self.instrument_wavelength_bin_widths = observatory_wavelength_bin_widths
        self.instrument_wavelength_bin_edges = observatory_wavelength_bin_edges
        self.grid_size = grid_size
        self.has_planet_orbital_motion = has_planet_orbital_motion
        self.modulation_period = modulation_period
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.observatory_coordinates = observatory_coordinates
        self.sources = sources
        self.simulation_time_step_length = simulation_time_step_length
        self.unperturbed_instrument_throughput = unperturbed_instrument_throughput
        self.instrument_time_steps = observatory_time_steps
        self.amplitude_perturbations = amplitude_perturbations
        self.phase_perturbations = phase_perturbations
        self.polarization_perturbations = polarization_perturbations
        self.simulation_time_steps = simulation_time_steps
        self.simulation_wavelength_bin_centers = simulation_wavelength_bin_centers
        self.simulation_wavelength_bin_widths = simulation_wavelength_bin_widths
        self.differential_output_pairs = differential_output_pairs

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

    def run(self) -> np.ndarray:
        """Run the data generator."""

        total_photon_counts = torch.zeros(
            (self.number_of_outputs, len(self.simulation_wavelength_bin_centers), len(self.simulation_time_steps)),
            device=self.device
        )

        for source in self.sources:
            total_photon_counts += calculate_photon_counts(
                self.device,
                self.aperture_radius,
                self.unperturbed_instrument_throughput,
                self.amplitude_perturbations,
                self.phase_perturbations,
                self.polarization_perturbations,
                self.observatory_coordinates[0],
                self.observatory_coordinates[1],
                source.sky_coordinates[0],
                source.sky_coordinates[1],
                source.sky_brightness_distribution,
                self.simulation_wavelength_bin_centers,
                self.simulation_wavelength_bin_widths,
                self.simulation_time_step_length,
                self.beam_combination_matrix,
                torch.empty(len(source.sky_brightness_distribution), device=self.device)
            )

        # Bin photon counts to observatory time and wavelengths
        time_binned_photon_counts = torch.asarray(
            block_reduce(
                total_photon_counts.cpu().numpy(),
                (1, 1, len(self.simulation_time_steps) // len(self.instrument_time_steps)),
                np.sum
            )
        ).to(self.device)

        binned_photon_counts = torch.zeros(
            (self.number_of_outputs, len(self.instrument_wavelength_bin_centers), time_binned_photon_counts.shape[2]),
            dtype=torch.float32,
            device=self.device
        )

        for index_wl, wl in enumerate(self.simulation_wavelength_bin_centers):
            index_closest_wavelength_edge = torch.abs(self.instrument_wavelength_bin_edges - wl).argmin()
            if index_closest_wavelength_edge == 0:
                index_wavelength_bin = 0
            elif wl <= self.instrument_wavelength_bin_edges[index_closest_wavelength_edge]:
                index_wavelength_bin = index_closest_wavelength_edge - 1
            else:
                index_wavelength_bin = index_closest_wavelength_edge

            binned_photon_counts[:, index_wavelength_bin, :] += time_binned_photon_counts[:, index_wl, :]

        # Calculate differential photon counts
        self.differential_photon_counts = torch.zeros(
            binned_photon_counts.shape,
            dtype=torch.float32,
            device=self.device
        )

        for index_pair, pair in enumerate(self.differential_output_pairs):
            self.differential_photon_counts[index_pair] = Tensor(
                binned_photon_counts[pair[0]] - binned_photon_counts[pair[1]]
            )

        return self.differential_photon_counts
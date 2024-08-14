import numpy as np
import torch
from torch import Tensor

from phringe.core.data_generator_helpers import calculate_complex_amplitude_base, \
    calculate_complex_amplitude, calculate_photon_counts_from_intensity_response
from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource


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
    :param differential_photon_counts: The differential photon counts
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
    """

    def __init__(
            self,
            aperture_radius: float,
            beam_combination_matrix: Tensor,
            differential_output_pairs: list[tuple[int, int]],
            detailed: bool,
            device: str,
            grid_size: int,
            has_planet_orbital_motion: bool,
            number_of_instrument_time_steps: float,
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
            unperturbed_instrument_throughput: Tensor
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
        self.detailed = detailed
        self.device = device
        self.instrument_wavelength_bin_centers = observatory_wavelength_bin_centers
        self.instrument_wavelength_bin_widths = observatory_wavelength_bin_widths
        self.instrument_wavelength_bin_edges = observatory_wavelength_bin_edges
        self.grid_size = grid_size
        self.has_planet_orbital_motion = has_planet_orbital_motion
        self.modulation_period = modulation_period
        self.number_of_differential_outputs = len(differential_output_pairs)
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.observatory_coordinates = observatory_coordinates
        self.sources = sources
        self.simulation_time_step_length = simulation_time_step_length
        self.unperturbed_instrument_throughput = unperturbed_instrument_throughput
        self.number_of_instrument_time_steps = number_of_instrument_time_steps
        self.amplitude_perturbations = amplitude_perturbations
        self.phase_perturbations = phase_perturbations
        self.polarization_perturbations = polarization_perturbations
        self.simulation_time_steps = simulation_time_steps
        self.simulation_wavelength_bin_centers = simulation_wavelength_bin_centers
        self.simulation_wavelength_bin_widths = simulation_wavelength_bin_widths
        self.differential_output_pairs = differential_output_pairs

    def run(self) -> np.ndarray:
        """Run the data generator."""

        total_photon_counts = torch.zeros(
            (self.number_of_outputs, len(self.simulation_wavelength_bin_centers), len(self.simulation_time_steps)),
            dtype=torch.float32,
            device=self.device
        )

        intensity_responses = {}

        for source in self.sources:
            # Calculate the complex amplitude (N_wavelengths x N_collectors x N_time_steps x N_pix x N_pix)
            base_complex_amplitude = calculate_complex_amplitude_base(
                self.amplitude_perturbations,
                self.phase_perturbations,
                self.observatory_coordinates[0],
                self.observatory_coordinates[1],
                source.sky_coordinates[0],
                source.sky_coordinates[1],
                self.simulation_wavelength_bin_centers
            ) * self.aperture_radius * torch.sqrt(self.unperturbed_instrument_throughput)

            complex_amplitude_x, complex_amplitude_y = calculate_complex_amplitude(
                base_complex_amplitude,
                self.polarization_perturbations
            )
            del base_complex_amplitude

            # Calculate the intensity response (N_wavelengths x N_outputs x N_time_steps x N_pix x N_pix)
            dot_product_x = (
                    self.beam_combination_matrix[None, ..., None, None, None] * complex_amplitude_x.unsqueeze(1)
            )
            del complex_amplitude_x

            result_x = torch.abs(torch.sum(dot_product_x, dim=2)) ** 2
            del dot_product_x

            dot_product_y = (
                    self.beam_combination_matrix[None, ..., None, None, None] * complex_amplitude_y.unsqueeze(1)
            )
            del complex_amplitude_y

            result_y = torch.abs(torch.sum(dot_product_y, dim=2)) ** 2
            del dot_product_y

            intensity_response = result_x + result_y

            if self.detailed:
                intensity_responses[source.name] = intensity_response

            # Calculate photon counts (N_outputs x N_wavelengths x N_time_steps)
            photon_counts = calculate_photon_counts_from_intensity_response(
                self.device,
                intensity_response,
                source.sky_brightness_distribution,
                self.simulation_wavelength_bin_centers,
                self.simulation_wavelength_bin_widths,
                self.simulation_time_step_length,
                torch.empty(len(source.sky_brightness_distribution), device=self.device)
            )

            total_photon_counts += photon_counts

        # Calculate differential photon counts (N_diff_outputs x N_spec_channels x N_time_steps)
        self.differential_photon_counts = torch.zeros(
            (
                self.number_of_differential_outputs,
                len(self.instrument_wavelength_bin_centers),
                len(self.simulation_time_steps)
            ),
            dtype=torch.float32,
            device=self.device
        )

        for index_pair, pair in enumerate(self.differential_output_pairs):
            self.differential_photon_counts[index_pair] = Tensor(
                total_photon_counts[pair[0]] - total_photon_counts[pair[1]]
            )

        return self.differential_photon_counts, intensity_responses

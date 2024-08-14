from copy import copy
from typing import Any

import numpy as np
import torch
from skimage.measure import block_reduce
from torch.cuda import OutOfMemoryError
from tqdm import tqdm

from phringe.core.data_generator import DataGenerator
from phringe.core.director_helpers import calculate_simulation_time_steps, calculate_nulling_baseline, \
    calculate_amplitude_perturbations, calculate_phase_perturbations, \
    calculate_polarization_perturbations, prepare_modeled_sources
from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.util.helpers import InputSpectrum


class Director():
    """Class representation of the director.

    :param amplitude_falloff_exponent: The amplitude falloff exponent
    :param amplitude_perturbation_rms: The amplitude perturbation RMS
    :param aperture_diameter: The aperture diameter
    :param array: The array
    :param baseline_maximum: The maximum baseline
    :param baseline_minimum: The minimum baseline
    :param baseline_ratio: The baseline ratio
    :param beam_combiner: The beam combiner
    :param beam_combination_transfer_matrix: The beam combination transfer matrix
    :param detailed: Whether to run in detailed mode
    :param detector_integration_time: The detector integration time
    :param device: The device
    :param differential_output_pairs: The differential output pairs
    :param grid_size: The grid size
    :param has_amplitude_perturbations: Whether to have amplitude perturbations
    :param has_exozodi_leakage: Whether to have exozodi leakage
    :param has_local_zodi_leakage: Whether to have local zodi leakage
    :param has_phase_perturbations: Whether to have phase perturbations
    :param has_planet_orbital_motion: Whether to have planet orbital motion
    :param has_planet_signal: Whether to have planet signal
    :param has_polarization_perturbations: Whether to have polarization perturbations
    :param has_stellar_leakage: Whether to have stellar leakage
    :param input_spectra: The input spectra
    :param instrument_wavelength_bin_centers: The instrument wavelength bin centers
    :param instrument_wavelength_bin_edges: The instrument wavelength bin edges
    :param instrument_wavelength_bin_widths: The instrument wavelength bin widths
    :param observatory_wavelength_range_lower_limit: The observatory wavelength range lower limit
    :param observatory_wavelength_range_upper_limit: The observatory wavelength range upper limit
    :param optimized_differential_output: The optimized differential output
    :param optimized_star_separation: The optimized star separation
    :param optimized_wavelength: The optimized wavelength
    :param phase_falloff_exponent: The phase falloff exponent
    :param phase_perturbation_rms: The phase perturbation RMS
    :param planets: The planets
    :param polarization_falloff_exponent: The polarization falloff exponent
    :param polarization_perturbation_rms: The polarization perturbation RMS
    :param solar_ecliptic_latitude: The solar ecliptic latitude
    :param sources: The sources
    :param star: The star
    :param time_step_size: The time step size
    :param total_integration_time: The total integration time
    :param unperturbed_instrument_throughput: The unperturbed instrument throughput
    """
    # _simulation_time_step_length = tensor(0.5, dtype=torch.float32)
    _maximum_simulation_wavelength_sampling = 1000

    def __init__(
            self,
            settings: Settings,
            observatory: Observatory,
            observation: Observation,
            scene: Scene,
            input_spectra: list[InputSpectrum],
            gpus: tuple[int] = None,
            detailed: bool = False,
            normalize: bool = False
    ):
        """Constructor method.

        :param settings: The settings
        :param observatory: The observatory
        :param observation: The observation
        :param scene: The scene
        :param input_spectra: The input spectra
        :param gpus: The GPUs
        :param detailed: Whether to run in detailed mode
        :param normalize: Whether to normalize the data to unit RMS along the time axis
        """
        self._amplitude_perturbation_lower_limit = observatory.amplitude_perturbation_lower_limit
        self._amplitude_perturbation_upper_limit = observatory.amplitude_perturbation_upper_limit
        self._aperture_diameter = observatory.aperture_diameter
        self._array = observatory.array
        self._baseline_maximum = observation.baseline_maximum
        self._baseline_minimum = observation.baseline_minimum
        self._baseline_ratio = observation.baseline_ratio
        self._beam_combiner = observatory.beam_combiner
        self._beam_combination_transfer_matrix = self._beam_combiner.get_beam_combination_transfer_matrix()
        self._detailed = detailed
        self._detector_integration_time = observation.detector_integration_time
        self._devices = self._get_devices(gpus)
        self._differential_output_pairs = self._beam_combiner.get_differential_output_pairs()
        self._gpus = gpus
        self._grid_size = settings.grid_size
        self._has_amplitude_perturbations = settings.has_amplitude_perturbations
        self._has_exozodi_leakage = settings.has_exozodi_leakage
        self._has_local_zodi_leakage = settings.has_local_zodi_leakage
        self._has_phase_perturbations = settings.has_phase_perturbations
        self._has_planet_orbital_motion = settings.has_planet_orbital_motion
        self._has_planet_signal = settings.has_planet_signal
        self._has_polarization_perturbations = settings.has_polarization_perturbations
        self._has_stellar_leakage = settings.has_stellar_leakage
        self._input_spectra = input_spectra
        self._modulation_period = observation.modulation_period
        self._normalize = normalize
        self._number_of_inputs = self._beam_combiner.number_of_inputs
        self._number_of_outputs = self._beam_combiner.number_of_outputs
        self._instrument_wavelength_bin_centers = observatory.wavelength_bin_centers
        self._instrument_wavelength_bin_edges = observatory.wavelength_bin_edges
        self._instrument_wavelength_bin_widths = observatory.wavelength_bin_widths
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
        self._time_step_size = settings.time_step_size
        self._total_integration_time = observation.total_integration_time
        self._unperturbed_instrument_throughput = observatory.unperturbed_instrument_throughput

    def _generate_sub_data2(self):
        # get available gpu memory
        total_memory = torch.cuda.get_device_properties(self._gpus[0]).total_memory
        free_memory = torch.cuda.memory_reserved(self._gpus[0]) + torch.cuda.memory_allocated(self._gpus[0])
        available_memory = total_memory - free_memory

        # get the size of the data that will be produced in the data generator
        self._run_

        # if the size of the data is larger than the available memory, divide the data into smaller parts that fit into the memory
        # loop over all parts, making sure to delete the parts after they are not used anymore to free up memory and don't pile it up
        # return the data

    def _generate_sub_data(self) -> tuple[list[torch.Tensor], list[dict]]:
        """Generate data in a memory-safe way. If GPU runs out of memory, divide the data along the time dimension and
        retry iteratively until no memory error occurs, then concatenate the data

        :return: The sub data and sub intensity response
        """
        # TODO: Implement a more sophisticated memory management
        divisor = 1
        is_successful = True
        while is_successful:
            if divisor > len(self.simulation_time_steps):
                raise OutOfMemoryError('Not enough memory to generate data. Choose other configurations.')

            sub_data = []
            sub_intensity_response = []
            indices = self._generate_sub_data_indices(divisor)

            # Generate data for each set of indices
            for i in tqdm(range(len(indices))):
                while i <= len(indices) - 2:
                    lower_index = indices[i]
                    upper_index = indices[i + 1]

                    try:
                        # Run data generator and append data and intensity response
                        data, intensity_response = self._run_data_generator(lower_index, upper_index, divisor)
                        sub_data.append(data)
                        sub_intensity_response.append(intensity_response)
                        is_successful = False
                        break

                    except OutOfMemoryError:
                        divisor *= 2
                        print(f'Out of memory error occurred. Dividing data into {divisor} parts and retrying.')
                        break

            if not len(sub_data) == len(indices) - 1:
                is_successful = True

        return sub_data, sub_intensity_response

    def _generate_sub_data_indices(self, divisor: int) -> list[int]:
        """Generate indices representing the data split into x parts, where x is given by 'divisor'.

        :param divisor: The divisor
        :return: The sub data indices
        """
        time_steps_per_iteration = len(self.simulation_time_steps) // divisor
        indices = []
        for i in range(0, divisor + 1):
            indices.append(i * time_steps_per_iteration)
        return indices

    def _get_devices(self, gpus: tuple[int]) -> list[str]:
        """Get the devices.

        :param gpus: The GPUs
        :return: The devices
        """
        devices = []
        if gpus and torch.cuda.is_available() and torch.cuda.device_count():
            if torch.max(torch.asarray(gpus)) > torch.cuda.device_count():
                raise ValueError(f'GPU number {torch.max(torch.asarray(gpus))} is not available on this machine.')
            for gpu in gpus:
                devices.append(torch.device(f'cuda:{gpu}'))
        else:
            devices.append(torch.device('cpu'))
        return devices

    def _run_data_generator(self, lower_index: int, upper_index: int, divisor: int) -> tuple[Any, Any]:
        """Run the data generator.

        :param lower_index: The lower index
        :param upper_index: The upper index
        :param divisor: The divisor
        :return: The data and intensity response
        """
        # If orbital motion is considered, planet sky coordinates and brightness distribution have a time axis as well,
        # i.e. change with time, and need to be sliced to the sub indices
        if self._has_planet_orbital_motion:
            planets_copy = []

            for index_planet, planet in enumerate(self._planets):
                planet_copy = copy(planet)
                planet_copy.sky_coordinates = copy(planet.sky_coordinates[:, lower_index:upper_index])
                planet_copy.sky_brightness_distribution = copy(
                    planet.sky_brightness_distribution[lower_index:upper_index])
                planets_copy.append(planet_copy)

            # Remove all planets from sources object and add new ones
            self._sources = [source for source in self._sources if not isinstance(source, Planet)]
            self._sources.extend(planets_copy)

        # Create and run the data generator object
        data_generator = DataGenerator(
            self._aperture_diameter / 2,
            self._beam_combination_transfer_matrix,
            self._differential_output_pairs,
            self._detailed,
            self._device,
            self._grid_size,
            self._has_planet_orbital_motion,
            len(self._instrument_time_steps) / divisor,
            self._instrument_wavelength_bin_centers,
            self._instrument_wavelength_bin_widths,
            self._instrument_wavelength_bin_edges,
            self._modulation_period,
            self._number_of_inputs,
            self._number_of_outputs,
            self.observatory_coordinates[:, :, lower_index:upper_index],
            self.amplitude_perturbations[:, lower_index:upper_index],
            self.phase_perturbations[:, lower_index:upper_index],
            self.polarization_perturbations[:, lower_index:upper_index],
            self._time_step_size,
            self.simulation_time_steps[lower_index:upper_index],
            self.simulation_wavelength_bin_centers,
            self.simulation_wavelength_bin_widths,
            self._sources,
            self._unperturbed_instrument_throughput
        )
        differential_photon_counts, intensity_responses = data_generator.run()

        import gc
        del data_generator

        gc.collect()

        return differential_photon_counts, intensity_responses

    def run(self):
        """Run the director. This includes the following steps:
        - Calculate simulation and instrument time steps
        - Calculate the simulation wavelength bins
        - Calculate field of view
        - Calculate the nulling baseline
        - Calculate the instrument perturbation time series
        - Calculate the observatory coordinates time series
        - Calculate the spectral flux densities, coordinates and brightness distributions of all sources in the scene
        - Move all tensors to the device (i.e. GPU, if available)
        - Generate the data in a memory-safe way
        - Bin the data to observatory time steps and wavelength steps

        """
        # Check simulation time step is smaller than detector integration time
        if self._time_step_size > self._detector_integration_time:
            raise ValueError('The simulation time step size must be smaller than the detector integration time.')

        # Calculate simulation and instrument time steps
        self.simulation_time_steps = calculate_simulation_time_steps(
            self._total_integration_time,
            self._time_step_size
        )
        self._instrument_time_steps = torch.linspace(
            0,
            self._total_integration_time,
            int(self._total_integration_time / self._detector_integration_time)
        )

        self.simulation_wavelength_bin_centers = self._instrument_wavelength_bin_centers
        self.simulation_wavelength_bin_widths = self._instrument_wavelength_bin_widths

        # Calculate field of view
        self.field_of_view = self.simulation_wavelength_bin_centers / self._aperture_diameter / 40

        # Calculate the nulling baseline
        self.nulling_baseline = calculate_nulling_baseline(
            self._star.habitable_zone_central_angular_radius,
            self._star.distance,
            self._optimized_star_separation,
            self._optimized_differential_output,
            self._optimized_wavelength,
            self._baseline_maximum,
            self._baseline_minimum,
            self._array.type.value,
            self._beam_combiner.type
        )

        # Calculate the instrument perturbations
        self.amplitude_perturbations = calculate_amplitude_perturbations(
            self._number_of_inputs,
            self.simulation_time_steps,
            self._amplitude_perturbation_lower_limit,
            self._amplitude_perturbation_upper_limit,
            self._has_amplitude_perturbations
        )
        self.phase_perturbations = calculate_phase_perturbations(
            self._number_of_inputs,
            self._detector_integration_time,
            self.simulation_time_steps,
            self._phase_perturbation_rms,
            self._phase_falloff_exponent,
            self._has_phase_perturbations
        )
        self.polarization_perturbations = calculate_polarization_perturbations(
            self._number_of_inputs,
            self._detector_integration_time,
            self.simulation_time_steps,
            self._polarization_perturbation_rms,
            self._polarization_falloff_exponent,
            self._has_polarization_perturbations
        )

        # Calculate the observatory coordinates
        self.observatory_coordinates = self._array.get_collector_coordinates(
            self.simulation_time_steps,
            self.nulling_baseline,
            self._modulation_period,
            self._baseline_ratio
        )

        # Calculate the spectral flux densities, coordinates and brightness distributions of all sources in the scene
        self._sources = prepare_modeled_sources(
            self._sources,
            self.simulation_time_steps,
            self.simulation_wavelength_bin_centers,
            self._input_spectra,
            self._grid_size,
            self.field_of_view,
            self._solar_ecliptic_latitude,
            self._has_planet_orbital_motion,
            self._has_planet_signal,
            self._has_stellar_leakage,
            self._has_local_zodi_leakage,
            self._has_exozodi_leakage
        )

        # Move all tensors to the device (i.e. GPU, if available)
        self._device = self._devices[0]
        self._aperture_diameter = self._aperture_diameter.to(self._device)
        self._beam_combination_transfer_matrix = self._beam_combination_transfer_matrix.to(self._device)
        self._instrument_time_steps = self._instrument_time_steps.to(self._device)
        self._instrument_wavelength_bin_centers = self._instrument_wavelength_bin_centers.to(self._device)
        self._instrument_wavelength_bin_widths = self._instrument_wavelength_bin_widths.to(self._device)
        self._instrument_wavelength_bin_edges = self._instrument_wavelength_bin_edges.to(self._device)
        self.observatory_coordinates = self.observatory_coordinates.to(self._device)
        self.amplitude_perturbations = self.amplitude_perturbations.to(self._device)
        self.phase_perturbations = self.phase_perturbations.to(self._device)
        self.polarization_perturbations = self.polarization_perturbations.to(self._device)
        self._time_step_size = self._time_step_size.to(self._device)
        self.simulation_time_steps = self.simulation_time_steps.to(self._device)
        self.simulation_wavelength_bin_centers = self.simulation_wavelength_bin_centers.to(self._device)
        self.simulation_wavelength_bin_widths = self.simulation_wavelength_bin_widths.to(self._device)

        for index_source, source in enumerate(self._sources):
            self._sources[index_source].spectral_flux_density = source.spectral_flux_density.to(self._device)
            self._sources[index_source].sky_coordinates = source.sky_coordinates.to(self._device)
            self._sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
                self._device)
        self._unperturbed_instrument_throughput = self._unperturbed_instrument_throughput.to(self._device)

        # Generate data
        sub_data, sub_intensity_response = self._generate_sub_data()

        # Concatenate data and intensity response
        concatenated_data = torch.cat(sub_data, dim=2).cpu()
        self._intensity_response = {
            key: torch.cat([dict[key].cpu() for dict in sub_intensity_response], dim=2)
            for key in sub_intensity_response[0]
        }

        # Bin data to from simulation time steps observatory/instrument time steps
        binning_factor = int(round(len(self.simulation_time_steps) / len(self._instrument_time_steps), 0))
        self._data = torch.asarray(
            block_reduce(
                concatenated_data.numpy(),
                (1, 1, binning_factor),
                np.sum
            )
        )

        # Normalize data (used for template creation)
        if self._normalize:
            self._data = torch.einsum('ijk, ij->ijk', self._data, 1 / torch.sqrt(torch.mean(self._data ** 2, axis=2)))

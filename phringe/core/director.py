from typing import Union

import numpy as np
import sympy
import torch
from skimage.measure import block_reduce
from sympy import Symbol, exp, I, pi, cos, sin, Abs, symbols
from torch import Tensor
from tqdm import tqdm

from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation_mode import ObservationMode
from phringe.core.entities.perturbations.amplitude_perturbation import AmplitudePerturbation
from phringe.core.entities.perturbations.phase_perturbation import PhasePerturbation
from phringe.core.entities.perturbations.polarization_perturbation import PolarizationPerturbation
from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.photon_sources.star import Star
from phringe.core.entities.scene import Scene
from phringe.core.entities.simulation import Simulation
from phringe.util.memory import get_available_memory


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
            simulation: Simulation,
            instrument: Instrument,
            observation_mode: ObservationMode,
            scene: Scene,
            seed: int = None,
            gpu: int = None,
            normalize: bool = False,
            detailed: bool = False,
            extra_memory: int = 1
    ):
        """Constructor method.

        :param simulation: The simulation
        :param instrument: The instrument
        :param observation_mode: The observation mode
        :param scene: The scene
        :param seed: The seed
        :param gpu: The GPU
        :param detailed: Whether to run in detailed mode
        :param normalize: Whether to normalize the data to unit RMS along the time axis
        :param extra_memory: Whether to use extra memory
        """
        self._aperture_diameter = instrument.aperture_diameter
        self._array_configuration_matrix = instrument.array_configuration_matrix
        self._baseline_maximum = instrument.baseline_maximum
        self._baseline_minimum = instrument.baseline_minimum
        self._complex_amplitude_transfer_matrix = instrument.complex_amplitude_transfer_matrix
        self._detailed = detailed
        self._detector_integration_time = observation_mode.detector_integration_time
        self._device = self._get_device(gpu)
        self._differential_outputs = instrument.differential_outputs
        self._extra_memory = extra_memory
        self._gpu = gpu
        self._grid_size = simulation.grid_size
        self._has_amplitude_perturbations = simulation.has_amplitude_perturbations
        self._has_exozodi_leakage = simulation.has_exozodi_leakage
        self._has_local_zodi_leakage = simulation.has_local_zodi_leakage
        self._has_phase_perturbations = simulation.has_phase_perturbations
        self._has_planet_orbital_motion = simulation.has_planet_orbital_motion
        self._has_planet_signal = simulation.has_planet_signal
        self._has_polarization_perturbations = simulation.has_polarization_perturbations
        self._has_stellar_leakage = simulation.has_stellar_leakage
        self._sep_at_max_mod_eff = instrument.sep_at_max_mod_eff
        self._modulation_period = observation_mode.modulation_period
        self._normalize = normalize
        self._number_of_inputs = self._complex_amplitude_transfer_matrix.shape[1]
        self._number_of_outputs = self._complex_amplitude_transfer_matrix.shape[0]
        self._optimized_differential_output = observation_mode.optimized_differential_output
        self._optimized_star_separation = observation_mode.optimized_star_separation
        self._optimized_wavelength = observation_mode.optimized_wavelength
        self._perturbations = instrument.get_all_perturbations()
        self._planets = scene.planets
        self._quantum_efficiency = instrument.quantum_efficiency
        self._seed = seed
        self._simulation_time_step_size = simulation.time_step_size
        self._solar_ecliptic_latitude = observation_mode.solar_ecliptic_latitude
        self._sources = scene.get_all_sources()
        self._star = scene.star
        self._throughput = instrument.throughput
        self._total_integration_time = observation_mode.total_integration_time
        self._wavelength_bin_centers = instrument.wavelength_bin_centers
        self._wavelength_bin_edges = instrument.wavelength_bin_edges
        self._wavelength_bin_widths = instrument.wavelength_bin_widths
        self._wavelength_range_lower_limit = instrument.wavelength_range_lower_limit
        self._wavelength_range_upper_limit = instrument.wavelength_range_upper_limit

    def _get_device(self, gpu: int) -> torch.device:
        """Get the device.

        :param gpu: The GPU
        :return: The device
        """
        if gpu and torch.cuda.is_available() and torch.cuda.device_count():
            if torch.max(torch.asarray(gpu)) > torch.cuda.device_count():
                raise ValueError(f'GPU number {torch.max(torch.asarray(gpu))} is not available on this machine.')
            device = torch.device(f'cuda:{gpu}')
        else:
            device = torch.device('cpu')
        return device

    def _get_nulling_baseline(
            self,
            star_habitable_zone_central_angular_radius: float,
            optimized_star_separation: Union[str, float],
            optimized_differential_output: int,
            optimized_wavelength: float,
            baseline_maximum: float,
            baseline_minimum: float,
            max_modulation_efficiency: list[float],
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

        nulling_baseline = (
                max_modulation_efficiency[optimized_differential_output]
                * optimized_wavelength
                / optimized_star_separation
        )

        if baseline_minimum <= nulling_baseline and nulling_baseline <= baseline_maximum:
            return nulling_baseline
        raise ValueError(
            f"Nulling baseline of {nulling_baseline} is not within allowed ranges of baselines {baseline_minimum}-{baseline_maximum}"
        )

    def _prepare_perturbations(self):
        """Prepare the perturbations.
        """
        amplitude_perturbation = \
            [perturbation for perturbation in self._perturbations if isinstance(perturbation, AmplitudePerturbation)][0]
        phase_perturbation = \
            [perturbation for perturbation in self._perturbations if isinstance(perturbation, PhasePerturbation)][0]
        polarization_perturbation = \
            [perturbation for perturbation in self._perturbations if
             isinstance(perturbation, PolarizationPerturbation)][0]

        if self._has_amplitude_perturbations:
            self.amplitude_pert_time_series = amplitude_perturbation.get_time_series(
                self._number_of_inputs,
                self._modulation_period,
                len(self.simulation_time_steps),
                self._seed
            )
        else:
            self.amplitude_pert_time_series = torch.ones(
                (self._number_of_inputs, len(self.simulation_time_steps)),
                dtype=torch.float32
            )

        if self._has_phase_perturbations:
            self.phase_pert_time_series = phase_perturbation.get_time_series(
                self._number_of_inputs,
                self._modulation_period,
                len(self.simulation_time_steps),
                self._seed,
                wavelengths=self._wavelength_bin_centers
            )
        else:
            self.phase_pert_time_series = torch.zeros(
                (self._number_of_inputs, len(self._wavelength_bin_centers), len(self.simulation_time_steps)),
                dtype=torch.float32
            )

        if self._has_polarization_perturbations:
            self.polarization_pert_time_series = polarization_perturbation.get_time_series(
                self._number_of_inputs,
                self._modulation_period,
                len(self.simulation_time_steps),
                self._seed
            )
        else:
            self.polarization_pert_time_series = torch.zeros(
                (self._number_of_inputs, len(self.simulation_time_steps)),
                dtype=torch.float32
            )

    def _prepare_sources(
            self,
            sources: list[BasePhotonSource],
            simulation_time_steps: Tensor,
            simulation_wavelength_bin_centers: Tensor,
            grid_size: int,
            field_of_view: Tensor,
            solar_ecliptic_latitude: float,
            has_planet_orbital_motion: bool,
            has_planet_signal: bool,
            has_stellar_leakage: bool,
            has_local_zodi_leakage: bool,
            has_exozodi_leakage: bool
    ) -> list[BasePhotonSource]:
        """Return the spectral flux densities, brightness distributions and coordinates for all sources in the scene.

        :param sources: The sources in the scene
        :param simulation_time_steps: The simulation time steps
        :param simulation_wavelength_bin_centers: The simulation wavelength bin centers
        :param grid_size: The grid size
        :param field_of_view: The field of view
        :param solar_ecliptic_latitude: The solar ecliptic latitude
        :param has_planet_orbital_motion: Whether the simulation has planet orbital motion
        :param has_stellar_leakage: Whether the simulation has stellar leakage
        :param has_local_zodi_leakage: Whether the simulation has local zodi leakage
        :param has_exozodi_leakage: Whether the simulation has exozodi leakage
        :return: The prepared sources
        """
        star = [star for star in sources if isinstance(star, Star)][0]
        planets = [planet for planet in sources if isinstance(planet, Planet)]
        local_zodi = [local_zodi for local_zodi in sources if isinstance(local_zodi, LocalZodi)][0]
        exozodi = [exozodi for exozodi in sources if isinstance(exozodi, Exozodi)][0]
        prepared_sources = []

        if has_planet_signal:
            for index_planet, planet in enumerate(planets):
                planet.prepare(
                    simulation_wavelength_bin_centers,
                    grid_size,
                    star_distance=star.distance,
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
        if self._simulation_time_step_size > self._detector_integration_time:
            raise ValueError('The simulation time step size must be smaller than the detector integration time.')

        ################################################################################################################
        # Symbolic expressions
        ################################################################################################################

        # Define symbols for symbolic expressions
        catm = self._complex_amplitude_transfer_matrix
        acm = self._array_configuration_matrix
        ex = {}
        ey = {}
        a = {}
        da = {}
        dphi = {}
        th = {}
        dth = {}
        t, tm, b, l, alpha, beta = symbols('t tm b l alpha beta')

        # Define complex amplitudes
        for k in range(self._number_of_inputs):
            a[k] = Symbol(f'a_{k}', real=True)
            da[k] = Symbol(f'da_{k}', real=True)
            dphi[k] = Symbol(f'dphi_{k}', real=True)
            th[k] = Symbol(f'th_{k}', real=True)
            dth[k] = Symbol(f'dth_{k}', real=True)
            ex[k] = a[k] * da[k] * exp(I * (2 * pi / l * (acm[0, k] * alpha + acm[1, k] * beta) + dphi[k])) * cos(
                th[k] + dth[k])
            ey[k] = a[k] * da[k] * exp(I * (2 * pi / l * (acm[0, k] * alpha + acm[1, k] * beta) + dphi[k])) * sin(
                th[k] + dth[k])

        # Define intensity response and save the symbolic expression
        r = {}
        rx = {}
        ry = {}
        r_torch = {}
        r_numpy = {}

        self._symbolic_intensity_response = {}
        for j in range(self._number_of_outputs):
            rx[j] = 0
            ry[j] = 0
            for k in range(self._number_of_inputs):
                rx[j] += catm[j, k] * ex[k]
                ry[j] += catm[j, k] * ey[k]
            r[j] = Abs(rx[j]) ** 2 + Abs(ry[j]) ** 2
            self._symbolic_intensity_response[j] = r[j]

        # Compile the intensity response functions for numerical calculations and save the lambdified functions
        def _torch_sqrt(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.sqrt(x)

        torch_func_dict = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'log': torch.log,
            'sqrt': _torch_sqrt
        }

        self._diff_ir_torch = {}
        self._diff_ir_numpy = {}

        for i in range(len(self._differential_outputs)):
            # Lambdify differential output for torch
            self._diff_ir_torch[i] = sympy.lambdify(
                [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
                r[self._differential_outputs[i][0]] - r[self._differential_outputs[i][1]],
                [torch_func_dict]
            )

            # Lambdify differential output for numpy
            self._diff_ir_numpy[i] = sympy.lambdify(
                [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
                r[self._differential_outputs[i][0]] - r[self._differential_outputs[i][1]],
                'numpy'
            )

        for j in range(self._number_of_outputs):
            # Lambdify intensity response for torch
            r[j] = sympy.lambdify(
                [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
                r[j],
                [torch_func_dict]
            )

        self._intensity_response_torch = r

        ################################################################################################################
        # Numerical calculations
        ################################################################################################################

        # Calculate simulation and detector time steps
        self.simulation_time_steps = torch.linspace(
            0,
            self._total_integration_time,
            int(self._total_integration_time / self._simulation_time_step_size)
        )
        self._detector_time_steps = torch.linspace(
            0,
            self._total_integration_time,
            int(self._total_integration_time / self._detector_integration_time)
        )

        # Calculate field of view
        self.field_of_view = self._wavelength_bin_centers / self._aperture_diameter

        # Calculate the nulling baseline
        self.nulling_baseline = self._get_nulling_baseline(
            self._star.habitable_zone_central_angular_radius,
            self._optimized_star_separation,
            self._optimized_differential_output,
            self._optimized_wavelength,
            self._baseline_maximum,
            self._baseline_minimum,
            self._sep_at_max_mod_eff
        )

        # Prepare perturbation time series
        self._prepare_perturbations()

        # Calculate the spectral flux densities, coordinates and brightness distributions of all sources in the scene
        self._sources = self._prepare_sources(
            self._sources,
            self.simulation_time_steps,
            self._wavelength_bin_centers,
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
        self._aperture_diameter = self._aperture_diameter.to(self._device)
        self._detector_time_steps = self._detector_time_steps.to(self._device)
        self._wavelength_bin_centers = self._wavelength_bin_centers.to(self._device)
        self._wavelength_bin_widths = self._wavelength_bin_widths.to(self._device)
        self._wavelength_bin_edges = self._wavelength_bin_edges.to(self._device)
        self.amplitude_pert_time_series = self.amplitude_pert_time_series.to(self._device)
        self.phase_pert_time_series = self.phase_pert_time_series.to(self._device)
        self.polarization_pert_time_series = self.polarization_pert_time_series.to(self._device)
        self._simulation_time_step_size = self._simulation_time_step_size.to(self._device)
        self.simulation_time_steps = self.simulation_time_steps.to(self._device)

        for index_source, source in enumerate(self._sources):
            self._sources[index_source].spectral_flux_density = source.spectral_flux_density.to(self._device)
            self._sources[index_source].sky_coordinates = source.sky_coordinates.to(self._device)
            self._sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
                self._device)

        # Calculate amplitude (assumed to be identical for each collector)
        self._amplitude = self._aperture_diameter / 2 * torch.sqrt(
            torch.tensor(self._throughput * self._quantum_efficiency, device=self._device))

        # Prepare the output tensors
        counts = torch.zeros(
            (self._number_of_outputs,
             len(self._wavelength_bin_centers),
             len(self.simulation_time_steps)),
            device=self._device
        )
        diff_counts = torch.zeros(
            (len(self._differential_outputs),
             len(self._wavelength_bin_centers),
             len(self.simulation_time_steps)),
            device=self._device
        )

        # Estimate the data size and slice the time steps to fit the calculations into memory
        data_size = (self._grid_size ** 2
                     * len(self.simulation_time_steps)
                     * len(self._wavelength_bin_centers)
                     * self._number_of_outputs
                     * 4  # should be 2, but only works with 6 so there you go
                     * len(self._sources))

        available_memory = get_available_memory(self._device) / self._extra_memory

        # Divisor with 10% safety margin
        divisor = int(np.ceil(data_size / (available_memory * 0.9)))

        time_step_indices = torch.arange(
            0,
            len(self.simulation_time_steps) + 1,
            len(self.simulation_time_steps) // divisor
        )

        # Add the last index if it is not already included due to rounding issues
        if time_step_indices[-1] != len(self.simulation_time_steps):
            time_step_indices = torch.cat((time_step_indices, torch.tensor([len(self.simulation_time_steps)])))

        # Calculate differential counts
        for index, it in tqdm(enumerate(time_step_indices), total=len(time_step_indices) - 1):

            # Calculate the indices of the time slices
            if index <= len(time_step_indices) - 2:
                it_low = it
                it_high = time_step_indices[index + 1]
            else:
                break

            for source in self._sources:

                # Broadcast sky coordinates to the correct shape
                if isinstance(source, LocalZodi) or isinstance(source, Exozodi):
                    sky_coordinates_x = source.sky_coordinates[0][:, None, :, :]
                    sky_coordinates_y = source.sky_coordinates[1][:, None, :, :]
                elif isinstance(source, Planet) and self._has_planet_orbital_motion:
                    sky_coordinates_x = source.sky_coordinates[0][None, it_low:it_high, :, :]
                    sky_coordinates_y = source.sky_coordinates[1][None, it_low:it_high, :, :]
                else:
                    sky_coordinates_x = source.sky_coordinates[0][None, None, :, :]
                    sky_coordinates_y = source.sky_coordinates[1][None, None, :, :]

                # Broadcast sky brightness distribution to the correct shape
                if isinstance(source, Planet) and self._has_planet_orbital_motion:
                    sky_brightness_distribution = source.sky_brightness_distribution.swapaxes(0, 1)
                else:
                    sky_brightness_distribution = source.sky_brightness_distribution[:, None, :, :]

                # Define normalization
                if isinstance(source, Planet):
                    normalization = 1
                elif isinstance(source, Star):
                    normalization = len(
                        source.sky_brightness_distribution[0][source.sky_brightness_distribution[0] > 0])
                else:
                    normalization = self._grid_size ** 2

                # Calculate counts of shape (N_outputs x N_wavelengths x N_time_steps) for all time step slices
                # Within torch.sum, the shape is (N_wavelengths x N_time_steps x N_pix x N_pix)
                for i in range(self._number_of_outputs):

                    # Calculate the counts of all outputs only in detailed mode. Else calculate only the ones needed to
                    # calculate the differential outputs
                    if not self._detailed and i not in np.array(self._differential_outputs).flatten():
                        continue

                    if self._seed is not None:
                        torch.manual_seed(self._seed + i)

                    counts[i, :, it_low:it_high] += torch.poisson((
                        torch.sum(
                            r[i](
                                self.simulation_time_steps[None, it_low:it_high, None, None],
                                self._wavelength_bin_centers[:, None, None, None],
                                sky_coordinates_x,
                                sky_coordinates_y,
                                torch.tensor(self._modulation_period, device=self._device),
                                torch.tensor(self.nulling_baseline, device=self._device),
                                *[self._amplitude for _ in range(self._number_of_inputs)],
                                *[self.amplitude_pert_time_series[k][None, it_low:it_high, None, None] for k in
                                  range(self._number_of_inputs)],
                                *[self.phase_pert_time_series[k][:, it_low:it_high, None, None] for k in
                                  range(self._number_of_inputs)],
                                *[torch.tensor(0, device=self._device) for _ in range(self._number_of_inputs)],
                                *[self.polarization_pert_time_series[k][None, it_low:it_high, None, None] for k in
                                  range(self._number_of_inputs)]
                            )
                            * sky_brightness_distribution
                            / normalization
                            * self._simulation_time_step_size
                            * self._wavelength_bin_widths[:, None, None, None], axis=(2, 3)
                        )
                    ))

        # Calculate differential outputs
        for i in range(len(self._differential_outputs)):
            diff_counts[i] = counts[self._differential_outputs[i][0]] - counts[self._differential_outputs[i][1]]

        # # Bin data to from simulation time steps detector time steps
        binning_factor = int(round(len(self.simulation_time_steps) / len(self._detector_time_steps), 0))

        if self._detailed:
            self._counts = torch.asarray(
                block_reduce(
                    counts.cpu().numpy(),
                    (1, 1, binning_factor),
                    np.sum
                )
            )

        self._data = torch.asarray(
            block_reduce(
                diff_counts.cpu().numpy(),
                (1, 1, binning_factor),
                np.sum
            )
        )
        #
        # # Normalize data (used for template creation)
        if self._normalize:
            self._data = torch.einsum('ijk, ij->ijk', self._data, 1 / torch.sqrt(torch.mean(self._data ** 2, axis=2)))

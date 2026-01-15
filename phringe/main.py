from pathlib import Path
from typing import Union, overload

import numpy as np
import torch
from astropy.constants.codata2018 import G
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from skimage.measure import block_reduce
from sympy import lambdify, symbols
from torch import Tensor
from tqdm import tqdm

from phringe.core.configuration import Configuration
from phringe.core.instrument import Instrument
from phringe.core.observation import Observation
from phringe.core.scene import Scene
from phringe.core.sources.exozodi import Exozodi
from phringe.core.sources.local_zodi import LocalZodi
from phringe.core.sources.planet import Planet
from phringe.core.sources.star import Star
from phringe.io.nifits_writer import NIFITSWriter
from phringe.processing.processing import get_sensitivity_limits, get_sep_at_max_mod_eff
from phringe.util.baseline import OptimalNullingBaseline
from phringe.util.device import get_available_memory
from phringe.util.device import get_device
from phringe.util.grid import get_meshgrid


class PHRINGE:
    """
    Main PHRINGE class.

    Parameters
    ----------
    seed : int or None
        Seed for the generation of random numbers. If None, a random seed is chosen.
    gpu_index : int or None
        Index corresponding to the GPU that should be used. If None or if the index is not available, the CPU is used.
    device : torch.device or None
        Device to use; alternatively to the index of the GPU. If None, the device is chosen based on the GPU index.
    grid_size : int
        Grid size used for the calculations.
    time_step_size : float
        Time step size used for the calculations. By default, this is the detector integration time. If it is smaller,
        the generated data will be rebinned to the detector integration times at the end of the calculations.
    extra_memory : int
        Extra memory factor to use for the calculations. This might be required to handle large data sets.

    Attributes
    ----------
    _detector_time_steps : torch.Tensor
        Detector time steps.
    _device : torch.device
        Device.
    _extra_memory : int
        Extra memory.
    _grid_size : int
        Grid size.
    _instrument : Instrument
        Instrument.
    _observation : Observation
        Observation.
    _scene : Scene
        Scene.
    _simulation_time_steps : torch.Tensor
        Simulation time steps.
    _time_step_size : float
        Time step size.
    seed : int
        Seed.
    """

    def __init__(
            self,
            seed: int = None,
            gpu_index: int = None,
            device: torch.device = None,
            grid_size=40,
            time_step_size: float = None,
            extra_memory: int = 1
    ):
        self._detector_time_steps = None
        self._device = get_device(gpu_index) if device is None else device
        self._extra_memory = extra_memory
        self._grid_size = grid_size
        self._instrument = None
        self._observation = None
        self._scene = None
        self._simulation_time_steps = None
        self._time_step_size = time_step_size

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    @property
    def detector_time_steps(self):
        return torch.linspace(
            0,
            self._observation.total_integration_time,
            int(self._observation.total_integration_time / self._observation.detector_integration_time),
            device=self._device
        ) if self._observation is not None else None

    @property
    def _simulation_time_step_size(self):
        if self._time_step_size is not None and self._time_step_size < self._observation.detector_integration_time:
            return self._time_step_size
        else:
            return self._observation.detector_integration_time

    @property
    def simulation_time_steps(self):
        return torch.linspace(
            0,
            self._observation.total_integration_time,
            int(self._observation.total_integration_time / self._simulation_time_step_size),
            device=self._device
        ) if self._observation is not None else None

    def _get_time_slices(self, ):
        """Estimate the data size and slice the time steps to fit the calculations into memory. This is necessary to
        avoid memory issues when calculating the counts for large data sets.

        """
        data_size = (self._grid_size ** 2
                     * len(self.simulation_time_steps)
                     * len(self._instrument.wavelength_bin_centers)
                     * self._instrument.number_of_outputs
                     * 4  # should be 2, but only works with 4 so there you go
                     * len(self._scene._get_all_sources()))

        available_memory = get_available_memory(self._device) / self._extra_memory

        # Divisor with 10% safety margin
        divisor = int(np.ceil(data_size / (available_memory * 0.9)))
        number_of_loops = len(self.simulation_time_steps) // divisor

        time_step_indices = torch.arange(
            0,
            len(self.simulation_time_steps) + 1,
            number_of_loops if number_of_loops > 0 else 1,
        )

        # Add the last index if it is not already included due to rounding issues
        if time_step_indices[-1] != len(self.simulation_time_steps):
            time_step_indices = torch.cat((time_step_indices, torch.tensor([len(self.simulation_time_steps)])))

        return time_step_indices

    def _get_unbinned_counts(self):
        """Calculate the differential counts for all time steps (, i.e. simulation time steps). Hence
        the output is not yet binned to detector time steps.

        """
        # if self.seed is not None: _set_seed(self.seed)

        # Prepare output tensor
        counts = torch.zeros(
            (self._instrument.number_of_outputs,
             len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            device=self._device
        )

        # Estimate the data size and slice the time steps to fit the calculations into memory
        time_step_indices = self._get_time_slices()

        # Calculate counts
        for index, it in tqdm(enumerate(time_step_indices), total=len(time_step_indices) - 1, disable=True):

            # Calculate the indices of the time slices
            if index <= len(time_step_indices) - 2:
                it_low = it
                it_high = time_step_indices[index + 1]
            else:
                break

            for source in self._scene._get_all_sources():

                # Broadcast sky coordinates to the correct shape
                if isinstance(source, LocalZodi) or isinstance(source, Exozodi):
                    sky_coordinates_x = source._sky_coordinates[0][:, None, :, :]
                    sky_coordinates_y = source._sky_coordinates[1][:, None, :, :]
                elif isinstance(source, Planet) and source.has_orbital_motion:
                    sky_coordinates_x = source._sky_coordinates[0][None, it_low:it_high, :, :]
                    sky_coordinates_y = source._sky_coordinates[1][None, it_low:it_high, :, :]
                else:
                    sky_coordinates_x = source._sky_coordinates[0][None, None, :, :]
                    sky_coordinates_y = source._sky_coordinates[1][None, None, :, :]

                # Broadcast sky brightness distribution to the correct shape
                if isinstance(source, Planet) and source.has_orbital_motion:
                    sky_brightness_distribution = source._sky_brightness_distribution.swapaxes(0, 1)[:, it_low:it_high,
                    :, :]
                else:
                    sky_brightness_distribution = source._sky_brightness_distribution[:, None, :, :]

                # Define normalization
                if isinstance(source, Planet):
                    normalization = 1
                elif isinstance(source, Star):
                    normalization = len(
                        source._sky_brightness_distribution[0][source._sky_brightness_distribution[0] > 0])
                else:
                    normalization = self._grid_size ** 2

                # Get perturbation time series
                n_in = self._instrument.number_of_inputs
                n_t = len(self.simulation_time_steps)
                n_wl = len(self._instrument.wavelength_bin_centers)
                amplitude_pert_time_series = self._instrument.amplitude_perturbation.time_series \
                    if self._instrument.amplitude_perturbation is not None \
                    else torch.zeros((n_in, n_t), dtype=torch.float32, device=self._device)
                phase_pert_time_series = self._instrument.phase_perturbation.time_series \
                    if self._instrument.phase_perturbation is not None \
                    else torch.zeros((n_in, n_wl, n_t), dtype=torch.float32, device=self._device)
                polarization_pert_time_series = self._instrument.polarization_perturbation.time_series \
                    if self._instrument.polarization_perturbation is not None \
                    else torch.zeros((n_in, n_t), dtype=torch.float32, device=self._device)

                # Calculate counts of shape (N_outputs x N_wavelengths x N_time_steps) for all time step slices
                # Within torch.sum, the shape is (N_wavelengths x N_time_steps x N_pix x N_pix)
                for i in range(self._instrument.number_of_outputs):
                    current_counts = (
                        torch.sum(
                            self._instrument.response[i](
                                self.simulation_time_steps[None, it_low:it_high, None, None],
                                self._instrument.wavelength_bin_centers[:, None, None, None],
                                sky_coordinates_x,
                                sky_coordinates_y,
                                torch.tensor(self._observation.modulation_period, device=self._device,
                                             dtype=torch.float32),
                                torch.tensor(self.get_nulling_baseline(), device=self._device, dtype=torch.float32),
                                *[self._instrument._get_amplitude(self._device) for _ in
                                  range(self._instrument.number_of_inputs)],
                                *[amplitude_pert_time_series[k][None, it_low:it_high, None, None] for k in
                                  range(self._instrument.number_of_inputs)],
                                *[phase_pert_time_series[k][:, it_low:it_high, None, None] for k in
                                  range(self._instrument.number_of_inputs)],
                                *[torch.tensor(0, device=self._device, dtype=torch.float32) for _ in
                                  range(self._instrument.number_of_inputs)],
                                *[polarization_pert_time_series[k][None, it_low:it_high,
                                None, None] for k in
                                  range(self._instrument.number_of_inputs)]
                            )
                            * sky_brightness_distribution
                            / normalization
                            * self._simulation_time_step_size
                            * self._instrument.wavelength_bin_widths[:, None, None, None], axis=(2, 3)
                        )
                    )
                    # Add photon (Poisson) noise
                    if self._device != torch.device('mps'):
                        current_counts = torch.poisson(current_counts)
                    else:
                        current_counts = torch.poisson(current_counts.cpu()).to(self._device)
                    counts[i, :, it_low:it_high] += current_counts

        # Bin data to from simulation time steps detector time steps
        binning_factor = int(round(len(self.simulation_time_steps) / len(self.detector_time_steps), 0))

        return counts, binning_factor

    def export_nifits(self, path: Path = Path('.'), filename: str = None, name_suffix: str = ''):
        NIFITSWriter().write(self, output_dir=path)

    def get_collector_positions(self):
        """Return the collector positions of the instrument as a tensor of shape (2 x N_inputs x N_time).

        Returns
        -------
        torch.Tensor
            Collector positions.
        """
        acm = self._instrument.array_configuration_matrix

        t, tm, b, q = symbols('t tm b q')
        acm_func = lambdify((t, tm, b, q), acm, modules='numpy')
        return acm_func(self.simulation_time_steps.cpu().numpy(), self._observation.modulation_period,
                        self.get_nulling_baseline(), 6)

    def get_counts(self, kernels: bool = False) -> Tensor:
        """Calculate and return the time-binned raw photoelectron counts for all outputs (N_outputs x N_wavelengths x N_time_steps)
        or for kernels (N_kernels x N_wavelengths x N_time_steps).

        Parameters
        ----------
        kernels : bool
            Whether to use kernels for the calculations. Default is True.

        Returns
        -------
        torch.Tensor
            Raw photoelectron counts.
        """
        counts_unbinned, binning_factor = self._get_unbinned_counts()

        if kernels:
            kernels_torch = torch.tensor(self._instrument.kernels.tolist(), dtype=torch.float32, device=self._device)
            counts_kernels_unbinned = torch.einsum('ij, jkl -> ikl', kernels_torch, counts_unbinned)

            return torch.asarray(
                block_reduce(
                    counts_kernels_unbinned.cpu().numpy(),
                    (1, 1, binning_factor),
                    np.sum
                ),
                dtype=torch.float32,
                device=self._device
            )

        return torch.asarray(
            block_reduce(
                counts_unbinned.cpu().numpy(),
                (1, 1, binning_factor),
                np.sum
            ),
            dtype=torch.float32,
            device=self._device
        )

    def get_field_of_view(self) -> Tensor:
        """Return the field of view.


        Returns
        -------
        torch.Tensor
            Field of view.
        """
        return self._instrument._field_of_view

    def get_spectral_covariance(self, amplitude_var, phase_var, polarization_var):
        times = self.simulation_time_steps[None, :, None, None]
        wavelength_bin_centers = self._instrument.wavelength_bin_centers[:, None, None, None]
        wavelength_bin_widths = self._instrument.wavelength_bin_widths[:, None, None, None]
        alpha = self._scene.star._sky_coordinates[0][None, None, :, :]
        beta = self._scene.star._sky_coordinates[1][None, None, :, :]
        star_sky_brightness_distribution = self._scene.star._sky_brightness_distribution[:, None, :, :]
        modulation_period = self._observation.modulation_period
        nulling_baseline = self.get_nulling_baseline()
        detector_integration_time = self._observation.detector_integration_time
        amplitudes = [self._instrument._get_amplitude(self._device) for _ in range(self._instrument.number_of_inputs)]
        perturbation_covariance = torch.diag(
            torch.ones(3 * self._instrument.number_of_inputs, dtype=torch.float32, device=self._device)
        )
        for i in range(3 * self._instrument.number_of_inputs):
            if i < self._instrument.number_of_inputs:
                perturbation_covariance[i, i] *= amplitude_var
            elif i < 2 * self._instrument.number_of_inputs:
                perturbation_covariance[i, i] *= phase_var
            else:
                perturbation_covariance[i, i] *= polarization_var

        # Get lambdified functions
        l_func, H_func = self._instrument._get_lambdified_spectral_covariance()

        # Get l and stack into tensor
        l = l_func(
            times,
            wavelength_bin_centers,
            alpha,
            beta,
            modulation_period,
            nulling_baseline,
            *amplitudes,
        )

        ref = None
        for row in l:
            for elem in row:
                if isinstance(elem, torch.Tensor):
                    ref = elem
                    break
            if ref is not None:
                break

        if ref is None:
            raise RuntimeError("No tensor found in output.")

        target_shape = ref.shape
        device = ref.device
        dtype = ref.dtype

        l_fixed = [
            [
                (elem if isinstance(elem, torch.Tensor)
                 else torch.full(target_shape, float(elem), device=device, dtype=dtype))
                for elem in row
            ]
            for row in l
        ]

        l = torch.stack([torch.stack(row, dim=0) for row in l_fixed], dim=0)

        # Calc tilde l
        tilde_l = self._instrument.quantum_efficiency * self._observation.detector_integration_time * \
                  self.get_wavelength_bin_widths()[None, None, :, None] * torch.sum(
            l * star_sky_brightness_distribution[None, None, :, :, :, :],
            dim=(-1, -2)
        )

        # Get H and stack into tensor
        H = H_func(
            times,
            wavelength_bin_centers,
            alpha,
            beta,
            modulation_period,
            nulling_baseline,
            *amplitudes,
        )

        ref = None
        for row in H:
            for row2 in row:
                for elem in row2:
                    if isinstance(elem, torch.Tensor):
                        ref = elem
                        break
            if ref is not None:
                break

        if ref is None:
            raise RuntimeError("No tensor found in output.")

        target_shape = ref.shape
        device = ref.device
        dtype = ref.dtype

        # print(target_shape)

        def ensure_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.zeros(target_shape, device=device, dtype=dtype)

        # First convert every single entry, THEN stack three levels
        H_fixed = [
            [
                [ensure_tensor(elem) for elem in row]  # m2
                for row in block  # m1
            ]
            for block in H  # j
        ]

        for j1, i1 in enumerate(H):
            for j2, i2 in enumerate(i1):
                for j3, i3 in enumerate(i2):
                    if not isinstance(i3, torch.Tensor):
                        H[j1][j2][j3] = torch.zeros(target_shape, device=device, dtype=dtype)

        def bla(x):
            if x.shape != target_shape:
                x = torch.zeros(target_shape, device=device, dtype=dtype)
            return x

        H_tensor = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([
                            bla(elem) for elem in row
                        ], dim=0)
                        for row in block  # stack m1
                    ],
                    dim=0
                )
                for block in H_fixed  # stack j
            ],
            dim=0
        )

        tilde_H = self._instrument.quantum_efficiency * self._observation.detector_integration_time * \
                  self.get_wavelength_bin_widths()[None, None, None, :, None] * torch.sum(
            H_tensor * star_sky_brightness_distribution[None, None, None, :, :, :, :],
            dim=(-1, -2)
        )

        # Get kernels
        kernels_torch = torch.tensor(self._instrument.kernels.tolist(), dtype=torch.float32, device=self._device)

        # diff_ir = torch.einsum('ij, jklmn -> iklmn', kernels_torch, ir)
        # print(tilde_l.shape)
        # print(tilde_H.shape)
        # print(kernels_torch.shape)

        tilde_l_kernel = tilde_l
        # tilde_l_kernel = torch.einsum('ij, jklm -> iklm', kernels_torch, tilde_l)
        tilde_H_kernel = tilde_H
        # tilde_H_kernel = torch.einsum('ij, jklmn -> iklmn', kernels_torch, tilde_H)

        tilde_l_reduced = tilde_l_kernel  # .mean(dim=(-1))
        Lambda_kernel = tilde_l_reduced.permute(0, 2, 1, 3)

        # print(tilde_l_reduced.shape)
        # print(Lambda_kernel.shape)

        tilde_H_reduced = tilde_H_kernel  # .mean(dim=(-1))
        shape = tilde_H_reduced.shape
        # print(shape)
        Gamma_kernel = tilde_H_reduced.reshape((shape[0], shape[1] ** 2, shape[3], shape[4])).permute(0, 2, 1, 3)

        # print(Gamma_kernel.shape)

        kron = torch.kron(perturbation_covariance, perturbation_covariance)

        # print(kron.shape)
        #
        # cov1 = (Lambda_kernel @ perturbation_covariance @ Lambda_kernel.transpose(1, 2) + \
        #         2 * Gamma_kernel @ kron @ Gamma_kernel.transpose(1, 2)).mean(dim=-1)

        # print(cov1)
        linear = torch.einsum(
            'o l m t, m n, o k n t -> o l k t',
            Lambda_kernel, perturbation_covariance, Lambda_kernel
        ).mean(dim=-1)  # -> (o, l, k)

        quad = torch.einsum(
            'o l a t, a b, o k b t -> o l k t',
            Gamma_kernel, kron, Gamma_kernel
        ).mean(dim=-1)
        quad = 2 * quad

        cov = linear + quad

        # print(cov1.shape)

        return cov

    def get_instrument_response(self, fov: float = 7.27e-7, kernels=False, perturbations=True) -> Tensor:
        """Get the empirical instrument response. This corresponds to an array of shape (n_out x n_wavelengths x
        n_time_steps x n_grid x n_grid) if kernels=False and (n_diff_out x n_wavelengths x n_time_steps x n_grid x
        n_grid) if kernels=True.

        Parameters
        ----------
        fov : float
            Field of view in rad for which to calculate the instrument response. Default is 7.27e-7 rad (150 mas).$

        kernels : bool
            Whether to use kernels for the calculations. Default is False.
        perturbations : bool
            Whether to include perturbations in the calculations. Default is True.

        Returns
        -------
        torch.Tensor
            Empirical instrument response.
        """
        times = self.simulation_time_steps[None, :, None, None]
        wavelengths = self._instrument.wavelength_bin_centers[:, None, None, None]
        x_coordinates, y_coordinates = get_meshgrid(
            fov,
            self._grid_size,
            self._device
        )
        x_coordinates = x_coordinates[None, None, :, :]
        y_coordinates = y_coordinates[None, None, :, :]

        amplitude_pert_time_series = self._instrument.amplitude_perturbation.time_series if (
                self._instrument.amplitude_perturbation is not None and perturbations) else torch.zeros(
            (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
            dtype=torch.float32,
            device=self._device
        )
        phase_pert_time_series = self._instrument.phase_perturbation.time_series if (
                self._instrument.phase_perturbation is not None and perturbations) else torch.zeros(
            (self._instrument.number_of_inputs, len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            dtype=torch.float32,
            device=self._device
        )
        polarization_pert_time_series = self._instrument.polarization_perturbation.time_series if (
                self._instrument.polarization_perturbation is not None and perturbations) else torch.zeros(
            (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
            dtype=torch.float32,
            device=self._device
        )

        ir = torch.stack([self._instrument.response[j](
            times,
            wavelengths,
            x_coordinates,
            y_coordinates,
            self._observation.modulation_period,
            self.get_nulling_baseline(),
            *[self._instrument._get_amplitude(self._device) for _ in range(self._instrument.number_of_inputs)],
            *[amplitude_pert_time_series[k][None, :, None, None] for k in
              range(self._instrument.number_of_inputs)],
            *[phase_pert_time_series[k][:, :, None, None] for k in
              range(self._instrument.number_of_inputs)],
            *[torch.tensor(0) for _ in range(self._instrument.number_of_inputs)],
            *[polarization_pert_time_series[k][None, :, None, None] for k in
              range(self._instrument.number_of_inputs)]
        ) for j in range(self._instrument.number_of_outputs)])

        if kernels:
            kernels_torch = torch.tensor(self._instrument.kernels.tolist(), dtype=torch.float32, device=self._device)
            diff_ir = torch.einsum('ij, jklmn -> iklmn', kernels_torch, ir)
            return diff_ir

        return ir

    @overload
    def get_model_counts(
            self,
            spectral_energy_distribution: np.ndarray,
            x_position: float,
            y_position: float,
            kernels: bool = False,
    ) -> np.ndarray:
        ...

    @overload
    def get_model_counts(
            self,
            spectral_energy_distribution: np.ndarray,
            semi_major_axis: float,
            eccentricity: float,
            inclination: float,
            raan: float,
            argument_of_periapsis: float,
            true_anomaly: float,
            host_star_distance: float,
            host_star_mass: float,
            planet_mass: float,
            kernels: bool = False,
    ) -> np.ndarray:
        ...

    def get_model_counts(self, spectral_energy_distribution: np.ndarray, kernels: bool = False, **kwargs) -> np.ndarray:
        """Return the planet template (model) counts for a given spectral energy distribution and  either 1) sky
        coordinates or 2) orbital elements.The output array has shape (n_diff_out x n_wavelengths x n_time_steps) if
        kernels=True and (n_out x n_wavelengths x n_time_steps) if kernels=False.

        Parameters
        ----------
        spectral_energy_distribution : numpy.ndarray
            Spectral energy distribution in photons/(m^2 s m).
        kernels : bool
            Whether to use kernels for the calculations. Default is False.
        **kwargs
            Either x_position and y_position (both float, in radians) or semi_major_axis (float, in meters), eccentricity
            (float), inclination (float, in radians), raan (float, in radians), argument_of_periapsis (float, in radians),
            true_anomaly (float, in radians), host_star_distance (float, in meters), host_star_mass (float, in kg) and planet_mass
            (float, in kg).

        Returns
        -------
        numpy.ndarray
            Model counts.
        """
        times = self.get_time_steps().cpu().numpy()
        wavelength_bin_centers = self.get_wavelength_bin_centers()[:, None, None, None].cpu().numpy()
        wavelength_bin_widths = self.get_wavelength_bin_widths()[None, :, None, None, None].cpu().numpy()
        amplitude = self._instrument._get_amplitude(self._device).cpu().numpy()

        if np.array(spectral_energy_distribution).ndim == 0:
            spectral_energy_distribution = np.array(spectral_energy_distribution)[None, None, None, None, None]
        else:
            spectral_energy_distribution = spectral_energy_distribution[None, :, None, None, None]

        # Check which overload is used
        if 'x_position' in kwargs and 'y_position' in kwargs:
            x_position = kwargs['x_position']
            y_position = kwargs['y_position']
            x_positions = np.array([x_position])[None, None, None, None] if x_position is not None else None
            y_positions = np.array([y_position])[None, None, None, None] if y_position is not None else None
            times = times[None, :, None, None]

        else:
            import astropy.units as u

            semi_major_axis = kwargs['semi_major_axis']
            eccentricity = kwargs['eccentricity']
            inclination = kwargs['inclination']
            raan = kwargs['raan']
            argument_of_periapsis = kwargs['argument_of_periapsis']
            true_anomaly = kwargs['true_anomaly']
            host_star_distance = kwargs['host_star_distance']
            host_star_mass = kwargs['host_star_mass']
            planet_mass = kwargs['planet_mass']

            star = Body(parent=None, k=G * (host_star_mass + planet_mass) * u.kg, name='Star')
            orbit = Orbit.from_classical(
                star,
                a=semi_major_axis * u.m,
                ecc=u.Quantity(eccentricity),
                inc=inclination * u.rad,
                raan=raan * u.rad,
                argp=argument_of_periapsis * u.rad,
                nu=true_anomaly * u.rad
            )

            x_positions = np.zeros(len(times))[None, :, None, None]
            y_positions = np.zeros(len(times))[None, :, None, None]

            for it, time in enumerate(times):
                orbit_propagated = orbit.propagate(time * u.s)
                x, y = (orbit_propagated.r[0].to(u.m).value, orbit_propagated.r[1].to(u.m).value)
                x_positions[:, it] = x / host_star_distance
                y_positions[:, it] = y / host_star_distance

            times = times[None, None, :, None, None]

        # Check if position is outside of field of view. If so, set factor to 0 to cancel the response
        fovs = self.get_field_of_view().cpu().numpy()
        factors = np.ones_like(fovs)
        x_pos = x_positions.item()
        y_pos = y_positions.item()
        for i, fov in enumerate(fovs):
            if x_pos > fov / 2 or y_pos > fov / 2:
                factors[i] = 0

        # Return the corresponding counts depending on kernel usage and photon noise inclusion
        if kernels:
            diff_ir = np.concatenate([self._instrument._diff_ir_numpy[i](
                times,
                wavelength_bin_centers,
                x_positions,
                y_positions,
                self._observation.modulation_period,
                self.get_nulling_baseline(),
                *[amplitude for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)]
            ) for i in range(self._instrument.kernels.shape[0])])

            diff_counts = (diff_ir
                           * spectral_energy_distribution
                           * self._observation.detector_integration_time
                           * wavelength_bin_widths
                           * factors[None, :, None, None, None]
                           )

            return diff_counts[:, :, :, 0, 0]
        else:
            ir = np.concatenate([self._instrument._ir_numpy[i](
                times,
                wavelength_bin_centers,
                x_positions,
                y_positions,
                self._observation.modulation_period,
                self.get_nulling_baseline(),
                *[amplitude for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)],
                *[0 for _ in range(self._instrument.number_of_inputs)]
            ) for i in range(self._instrument.number_of_outputs)])

            counts = (ir
                      * spectral_energy_distribution
                      * self._observation.detector_integration_time
                      * wavelength_bin_widths
                      * factors[None, :, None, None, None]
                      )
            return counts[:, :, :, 0, 0]

    def get_null_depth(self) -> Tensor:
        """Return the null depth as an array of shape (n_diff_out x n_wavelengths x n_time_steps).


        Returns
        -------
        torch.Tensor
            Null depth.
        """
        if self._scene.star is None:
            raise ValueError('Null depth can only be calculated for a scene with a star.')

        star_sky_brightness = self._scene.star._sky_brightness_distribution
        star_sky_coordiantes = self._scene.star._sky_coordinates

        x_max = star_sky_coordiantes[0].max()
        diff_ir_emp = self.get_instrument_response(fov=2 * abs(x_max), kernels=True, perturbations=True)
        imax = torch.sum(star_sky_brightness, dim=(1, 2))
        imin = torch.sum(diff_ir_emp @ star_sky_brightness[None, :, None, :, :], dim=(3, 4))
        null = abs(imin / imax[None, :, None])
        return null

    def get_nulling_baseline(self) -> float:
        """Return the nulling baseline. If it has not been set manually, it is calculated using the observation and instrument parameters.


        Returns
        -------
        float
            Nulling baseline.

        Returns
        -------
        torch.Tensor
            Indices of the time slices.
        """
        return self._observation._nulling_baseline

    def get_sensitivity_limits(
            self,
            temperature: float,
            pfa: float = 2.9e-7,
            pdet: float = 0.9,
            ang_seps_mas: Union[list, np.ndarray, torch.tensor] = np.linspace(10, 150, 2),
            num_reps: int = 1,
            as_radius: bool = True,
            diag_only: bool = False,
    ) -> Tensor:
        """Return the sensitivity limits of the instrument. Returns inf if the planet is outside the fov.


        Returns
        -------
        torch.Tensor
            Sensitivity limits.
        """
        return get_sensitivity_limits(
            get_counts=self.get_counts,
            get_model_counts=self.get_model_counts,
            wavelength_bin_centers=self.get_wavelength_bin_centers(),
            scene=self._scene,
            device=self._device,
            temperature=temperature,
            pfa=pfa,
            pdet=pdet,
            ang_seps_mas=ang_seps_mas,
            num_reps=num_reps,
            as_radius=as_radius,
            diag_only=diag_only,
        )

    def get_sep_at_max_mod_eff(self, optimal_nulling_baseline: OptimalNullingBaseline) -> Union[float, tuple]:
        """Return the separation at maximum modulation efficiency in units of (optimized wavelength / nulling baseline).

        Parameters
        ----------
        optimal_nulling_baseline : OptimalNullingBaseline
            Optimal nulling baseline object to extract the optimized wavelength that was used in the setup.

        Returns
        -------
        torch.Tensor
            Separation at maximum modulation efficiency.
        """
        return get_sep_at_max_mod_eff(
            current_nulling_baseline=self.get_nulling_baseline(),
            optimal_nulling_baseline=optimal_nulling_baseline,
            get_instrument_response=self.get_instrument_response,
            wavelength_bin_centers=self.get_wavelength_bin_centers()
        )

    def get_source_spectrum(self, source_name: str) -> Tensor:
        """Return the spectral energy distribution of a source.

        Parameters
        ----------
        source_name : str
            Name of the source.

        Returns
        -------
        torch.Tensor
            Spectral energy distribution of the source.
        """
        return self._scene._get_source(source_name)._spectral_energy_distribution

    def get_time_steps(self) -> Tensor:
        """Return the detector time steps.


        Returns
        -------
        torch.Tensor
            Detector time steps.
        """

        return self.detector_time_steps

    def get_wavelength_bin_centers(self) -> Tensor:
        """Return the wavelength bin centers.


        Returns
        -------
        torch.Tensor
            Wavelength bin centers.
        """
        return self._instrument.wavelength_bin_centers

    def get_wavelength_bin_edges(self) -> Tensor:
        """Return the wavelength bin edges.


        Returns
        -------
        torch.Tensor
            Wavelength bin edges.
        """
        return self._instrument.wavelength_bin_edges

    def get_wavelength_bin_widths(self) -> Tensor:
        """Return the wavelength bin widths.


        Returns
        -------
        torch.Tensor
            Wavelength bin widths.
        """
        return self._instrument.wavelength_bin_widths

    def set(self, entity: Union[Instrument, Observation, Scene, Configuration]):
        """Set the instrument, observation, scene, or configuration.

        Parameters
        ----------
        entity : Instrument or Observation or Scene or Configuration
            Instrument, observation, scene, or configuration.
        """
        entity._phringe = self
        if isinstance(entity, Instrument):
            self._instrument = entity
        elif isinstance(entity, Observation):
            self._observation = entity
        elif isinstance(entity, Scene):
            self._scene = entity
        elif isinstance(entity, Configuration):
            self._observation = Observation(**entity.config_dict['observation'], _phringe=self)
            self._instrument = Instrument(**entity.config_dict['instrument'], _phringe=self)
            self._scene = Scene(**entity.config_dict['scene'], _phringe=self)
        else:
            raise ValueError(f'Invalid entity type: {type(entity)}')

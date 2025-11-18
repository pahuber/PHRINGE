from pathlib import Path
from typing import Union, overload

import astropy.units as u
import numpy as np
import torch
from astropy.constants.codata2018 import G
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from skimage.measure import block_reduce
from sympy import lambdify, symbols
from torch import Tensor
from torch.distributions import Normal
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
from phringe.util.device import get_available_memory
from phringe.util.device import get_device
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_standard_units


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

        time_step_indices = torch.arange(
            0,
            len(self.simulation_time_steps) + 1,
            len(self.simulation_time_steps) // divisor
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
        """Return the collector positions of the instrument as a tensor of shape (N_inputs x 2).

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

    def get_instrument_response(self, fov: float = None, kernels=False, perturbations=True) -> Tensor:
        """Get the empirical instrument response. This corresponds to an array of shape (n_out x n_wavelengths x
        n_time_steps x n_grid x n_grid) if kernels=False and (n_diff_out x n_wavelengths x n_time_steps x n_grid x
        n_grid) if kernels=True.

        Parameters
        ----------
        fov : float
            Field of view for which to calculate the instrument response. If None, the instrument's field of view is used.
        kernels : bool
            Whether to use kernels for the calculations. Default is False.
        perturbations : bool
            Whether to include perturbations in the calculations. Default is True.

        Returns
        -------
        torch.Tensor
            Empirical instrument response.
        """
        if fov is not None:
            fov = torch.tensor(fov, device=self._device)
        times = self.simulation_time_steps[None, :, None, None]
        wavelengths = self._instrument.wavelength_bin_centers[:, None, None, None]
        x_coordinates, y_coordinates = get_meshgrid(
            torch.max(fov if fov is not None else self._instrument._field_of_view),
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

    def get_model_counts(
            self,
            spectral_energy_distribution: np.ndarray,
            kernels: bool = False,
            **kwargs,
    ) -> np.ndarray:
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

            diff_counts = diff_ir * spectral_energy_distribution * self._observation.detector_integration_time * wavelength_bin_widths

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

            counts = ir * spectral_energy_distribution * self._observation.detector_integration_time * wavelength_bin_widths
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
            ang_sep_mas_min: float = 10,
            ang_sep_mas_max: float = 150,
            num_ang_seps: int = 10,
            num_reps: int = 1,
            as_radius: bool = True,
            make_2d: bool = False
    ) -> Tensor:
        """Return the sensitivity limits of the instrument.


        Returns
        -------
        torch.Tensor
            Sensitivity limits.
        """
        ang_seps_mas = np.linspace(ang_sep_mas_min, ang_sep_mas_max, num_ang_seps)
        ang_seps_rad = ang_seps_mas * (1e-3 / 3600) * (np.pi / 180)
        sensitivities = torch.zeros((num_reps, num_ang_seps), device=self._device)

        for i, ang_sep in enumerate(tqdm(ang_seps_rad)):
            for rep in range(num_reps):
                # Get whitening matrix
                n_ref = self.get_counts(kernels=True)
                n_ref = n_ref.permute(1, 0, 2).reshape(n_ref.shape[1], -1)

                cov = torch.cov(n_ref)
                # eigvals, eigvecs = torch.linalg.eigh(cov)
                # w = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + 1e-8)) @ eigvecs.T
                eigvals, eigvecs = torch.linalg.eigh(cov)
                w = eigvecs @ torch.diag(eigvals.clamp(min=1e-12).rsqrt()) @ eigvecs.T

                # Get model
                solid_angle_ref = 1e-20

                x0 = self.get_model_counts(
                    spectral_energy_distribution=get_blackbody_spectrum_standard_units(
                        temperature,
                        self.get_wavelength_bin_centers()
                    ).cpu().numpy(),
                    x_position=ang_sep,
                    y_position=0,
                    kernels=True
                )
                x0 = x0.transpose(1, 0, 2).reshape(x0.shape[1], -1) * solid_angle_ref
                x0 = torch.from_numpy(x0).float().to(n_ref.device)

                # Whiten model
                xw = w @ x0
                xw = xw.flatten()
                s = torch.linalg.norm(xw)

                # Calculate sensitivity limit
                std_normal = Normal(0.0, 1.0)
                zfa = std_normal.icdf(torch.tensor(1.0 - pfa, device=s.device))
                zdet = std_normal.icdf(torch.tensor(pdet, device=s.device))
                omega_min = solid_angle_ref * (zfa - zdet) / s  # minimal solid angle (sr) to hit (pfa,pdet)

                sensitivities[rep, i] = omega_min

        if as_radius:
            sensitivities = self._scene.star.distance * torch.sqrt(sensitivities / torch.pi) / (1 * u.Rearth).to(
                u.m).value
            # if make_2d:
        #     profile = radii.cpu().numpy()
        #     N = 2 * len(radii)  # image size (NxN)
        #     cx = cy = (N - 1) / 2
        #
        #     # pixel radii
        #     y, x = np.ogrid[:N, :N]
        #     r = np.hypot(x - cx, y - cy)
        #
        #     # map radii to profile indices
        #     # assume profile is uniformly sampled in radius from 0 to r_max
        #     r_max = r.max()
        #     r_profile = np.linspace(0, r_max, profile.size)
        #
        #     # fast linear interpolation
        #     img = np.interp(r, r_profile, profile, left=profile[0], right=profile[-1])
        #
        #     # optional: smoother interpolation
        #     # f = interp1d(r_profile, profile, kind='cubic', bounds_error=False,
        #     #              fill_value=(profile[0], profile[-1]))
        #     # img = f(r)
        #
        #     plt.imshow(img, origin='lower', cmap='viridis')
        #     plt.colorbar()
        #     plt.show()
        #
        #     return None

        return sensitivities

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

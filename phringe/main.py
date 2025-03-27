from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy import ndarray
from skimage.measure import block_reduce
from torch import Tensor
from tqdm import tqdm

from phringe.entities.configuration import Configuration
from phringe.entities.instrument import Instrument
from phringe.entities.observation import Observation
from phringe.entities.scene import Scene
from phringe.entities.sources.exozodi import Exozodi
from phringe.entities.sources.local_zodi import LocalZodi
from phringe.entities.sources.planet import Planet
from phringe.entities.sources.star import Star
from phringe.io.fits_writer import FITSWriter
from phringe.util.grid import get_meshgrid
from phringe.util.memory import get_available_memory


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

    Attributes
    ----------
    _detailed : bool
        Detailed.
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
    _simulation_time_step_size : float
        Simulation time step size.
    _simulation_time_steps : torch.Tensor
        Simulation time steps.
    _time_step_size : float
        Time step size.
    _normalize : bool
        Normalize.
    detector_time_steps : torch.Tensor
        Detector time steps.
    seed : int
        Seed.
    simulation_time_steps : torch.Tensor
        Simulation time steps.
    """

    def __init__(
            self,
            seed: int = None,
            gpu_index: int = None,
            device: torch.device = None,
            grid_size=40,
            time_step_size: float = None
    ):
        self._detailed = False  # TODO: implement correctly
        self._detector_time_steps = None
        self._device = self._get_device(gpu_index) if device is None else device
        self._extra_memory = 1
        self._grid_size = grid_size
        self._instrument = None
        self._normalize = False
        self._observation = None
        self._scene = None
        self._simulation_time_steps = None
        self._time_step_size = time_step_size
        self.seed = seed

        self._set_seed(self.seed if self.seed is not None else np.random.randint(0, 2 ** 31 - 1))

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

    def _get_unbinned_counts(self, diff_only: bool = False):
        """Calculate the differential counts for all time steps (, i.e. simulation time steps). Hence
        the output is not yet binned to detector time steps.

        """
        if self.seed is not None: self._set_seed(self.seed)
        # Prepare output tensor
        counts = torch.zeros(
            (self._instrument.number_of_outputs,
             len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            device=self._device
        )

        # Estimate the data size and slice the time steps to fit the calculations into memory
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

        # Calculate counts
        for index, it in tqdm(enumerate(time_step_indices), total=len(time_step_indices) - 1):

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
                    sky_brightness_distribution = source._sky_brightness_distribution.swapaxes(0, 1)
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

                # Calculate counts of shape (N_outputs x N_wavelengths x N_time_steps) for all time step slices
                # Within torch.sum, the shape is (N_wavelengths x N_time_steps x N_pix x N_pix)
                for i in range(self._instrument.number_of_outputs):

                    # Calculate the counts of all outputs only in detailed mode. Else calculate only the ones needed to
                    # calculate the differential outputs
                    if not diff_only and i not in np.array(self._instrument.differential_outputs).flatten():
                        continue

                    if self._normalize:
                        sky_brightness_distribution[sky_brightness_distribution > 0] = 1

                    current_counts = (
                        torch.sum(
                            self._instrument.response[i](
                                self.simulation_time_steps[None, it_low:it_high, None, None],
                                self._instrument.wavelength_bin_centers[:, None, None, None],
                                sky_coordinates_x,
                                sky_coordinates_y,
                                torch.tensor(self._observation.modulation_period, device=self._device),
                                torch.tensor(self._instrument._nulling_baseline, device=self._device),
                                *[self._instrument._get_amplitude(self._device) for _ in
                                  range(self._instrument.number_of_inputs)],
                                *[self._instrument.perturbations.amplitude._time_series[k][None, it_low:it_high, None,
                                  None] for k in
                                  range(self._instrument.number_of_inputs)],
                                *[self._instrument.perturbations.phase._time_series[k][:, it_low:it_high, None, None]
                                  for k in
                                  range(self._instrument.number_of_inputs)],
                                *[torch.tensor(0, device=self._device) for _ in
                                  range(self._instrument.number_of_inputs)],
                                *[self._instrument.perturbations.polarization._time_series[k][None, it_low:it_high,
                                  None, None] for k in
                                  range(self._instrument.number_of_inputs)]
                            )
                            * sky_brightness_distribution
                            / normalization
                            * self._simulation_time_step_size
                            * self._instrument.wavelength_bin_widths[:, None, None, None], axis=(2, 3)
                        )
                    )
                    if not self._normalize:
                        current_counts = torch.poisson(current_counts)

                    counts[i, :, it_low:it_high] += current_counts

        # Bin data to from simulation time steps detector time steps
        binning_factor = int(round(len(self.simulation_time_steps) / len(self.detector_time_steps), 0))

        return counts, binning_factor

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def export_nifits(self, data: Tensor, path: Path = Path('.'), filename: str = None, name_suffix: str = ''):
        FITSWriter().write(data, path, name_suffix)

    def get_counts(self) -> Tensor:
        """Calculate and return the raw photoelectron counts as a tensor of shape (N_outputs x N_wavelengths x N_time_steps).


        Returns
        -------
        torch.Tensor
            Raw photoelectron counts.
        """
        # Move all tensors to the device
        # self._instrument.aperture_diameter = self._instrument.aperture_diameter.to(self._device)

        counts, binning_factor = self._get_unbinned_counts(diff_only=True)

        counts = torch.asarray(
            block_reduce(
                counts.cpu().numpy(),
                (1, 1, binning_factor),
                np.sum
            )
        )

        return counts

    def get_diff_counts(self) -> Tensor:
        """Calculate and return the differential photoelectron counts as a tensor of shape (N_differential_outputs x N_wavelengths x N_time_steps).


        Returns
        -------
        torch.Tensor
            Differential photoelectron counts.
        """
        diff_counts = torch.zeros(
            (len(self._instrument.differential_outputs),
             len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            device=self._device
        )

        counts, binning_factor = self._get_unbinned_counts(diff_only=True)

        # Calculate differential outputs
        for i in range(len(self._instrument.differential_outputs)):
            diff_counts[i] = counts[self._instrument.differential_outputs[i][0]] - counts[
                self._instrument.differential_outputs[i][1]]

        diff_counts = torch.asarray(
            block_reduce(
                diff_counts.cpu().numpy(),
                (1, 1, binning_factor),
                np.sum
            )
        )

        return diff_counts

    def get_instrument_response(
            self,
            output: int,
            times: Union[float, ndarray, Tensor],
            wavelengths: Union[float, ndarray, Tensor],
            field_of_view: float,
            nulling_baseline: float = None,
            output_as_numpy: bool = False,
    ):
        if self.seed is not None: self._set_seed(self.seed)

        # Handle broadcasting and type conversions
        if isinstance(times, ndarray) or isinstance(times, float) or isinstance(times, int):
            times = torch.tensor(times, device=self._device)
        times = times[None, None, None, None]

        if isinstance(wavelengths, ndarray) or isinstance(wavelengths, float) or isinstance(wavelengths, int):
            wavelengths = torch.tensor(wavelengths, device=self._device)
        wavelengths = wavelengths[None, None, None, None]

        x_coordinates, y_coordinates = get_meshgrid(field_of_view, self._grid_size, self._device)
        x_coordinates = x_coordinates.to(self._device)
        y_coordinates = y_coordinates.to(self._device)
        x_coordinates = x_coordinates[None, None, :, :]
        y_coordinates = y_coordinates[None, None, :, :]

        times = self.simulation_time_steps if times is None else times
        wavelengths = self._instrument.wavelength_bin_centers if wavelengths is None else wavelengths
        x_coordinates = self._scene.star._sky_coordinates[0] if x_coordinates is None else x_coordinates
        y_coordinates = self._scene.star._sky_coordinates[1] if y_coordinates is None else y_coordinates

        # Calculate perturbation time series unless they have been manually set by the user. If no seed is set, the time
        # series are different every time this method is called
        amplitude_pert_time_series = self._instrument.perturbations.amplitude._time_series if self._instrument.perturbations.amplitude is not None else torch.zeros(
            (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
            dtype=torch.float32
        )
        phase_pert_time_series = self._instrument.perturbations.phase._time_series if self._instrument.perturbations.phase is not None else torch.zeros(
            (self._instrument.number_of_inputs, len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            dtype=torch.float32
        )
        polarization_pert_time_series = self._instrument.perturbations.polarization._time_series if self._instrument.perturbations.polarization is not None else torch.zeros(
            (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
            dtype=torch.float32
        )

        response = torch.stack([self._instrument.response[output](
            times,
            wavelengths,
            x_coordinates,
            y_coordinates,
            self._observation.modulation_period,
            nulling_baseline,
            *[self._instrument._get_amplitude(self._device) for _ in range(self._instrument.number_of_inputs)],
            *[amplitude_pert_time_series[k][None, :, None, None] for k in
              range(self._instrument.number_of_inputs)],
            *[phase_pert_time_series[k][:, :, None, None] for k in
              range(self._instrument.number_of_inputs)],
            *[torch.tensor(0) for _ in range(self._instrument.number_of_inputs)],
            *[polarization_pert_time_series[k][None, :, None, None] for k in
              range(self._instrument.number_of_inputs)]
        ) for j in range(self._instrument.number_of_outputs)])

        if output_as_numpy:
            return response.cpu().numpy()

        return response

    def get_nulling_baseline(self) -> float:
        """Return the nulling baseline. If it has not been set manually, it is calculated using the observation and instrument parameters.


        Returns
        -------
        float
            Nulling baseline.

        """
        return self._instrument._nulling_baseline

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

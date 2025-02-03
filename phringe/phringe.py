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
from phringe.util.grid import get_meshgrid
from phringe.util.memory import get_available_memory


class PHRINGE:
    """
    Main PHRINGE class.
    """

    def __init__(
            self,
            seed: int = None,
            gpu: int = None,
            device: torch.device = None,
            grid_size=40,
            time_step_size: float = 600  # TODO: imeplement this
    ):
        self._device = self._get_device(gpu) if device is None else device
        self._instrument = None
        self._observation = None
        self._scene = None
        self.seed = seed
        self._grid_size = grid_size
        self._simulation_time_step_size = time_step_size
        self._simulation_time_steps = None
        self._detector_time_steps = None
        self._detailed = False
        self._normalize = False
        self._extra_memory = 1

    @property
    def detector_time_steps(self):
        return torch.linspace(
            0,
            self._observation.total_integration_time,
            int(self._observation.total_integration_time / self._observation.detector_integration_time),
            device=self._device
        ) if self._observation is not None else None

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
                     * len(self._scene.get_all_sources()))

        available_memory = get_available_memory(self._device) / self._extra_memory

        # Divisor with 10% safety margin
        divisor = int(np.ceil(data_size / (available_memory * 0.9)))

        time_step_indices = torch.arange(
            0,
            len(self.simulation_time_steps) + 1,
            len(self.simulation_time_steps) // divisor
        )

        nulling_baseline = 14  # TODO: implement correctly

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

            for source in self._scene.get_all_sources():

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
                                torch.tensor(nulling_baseline, device=self._device),
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

    def get_counts(self):
        # Move all tensors to the device
        self._instrument.aperture_diameter = self._instrument.aperture_diameter.to(self._device)

        counts, binning_factor = self._get_unbinned_counts(diff_only=True)

        counts = torch.asarray(
            block_reduce(
                counts.cpu().numpy(),
                (1, 1, binning_factor),
                np.sum
            )
        )

        return counts

    def get_diff_counts(self):

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

        # Handle broadcasting and type conversions
        if isinstance(times, ndarray) or isinstance(times, float) or isinstance(times, int):
            times = torch.tensor(times, device=self._device)
        times = times[None, None, None, None]

        if isinstance(wavelengths, ndarray) or isinstance(wavelengths, float) or isinstance(wavelengths, int):
            wavelengths = torch.tensor(wavelengths, device=self._device)
        wavelengths = wavelengths[None, None, None, None]

        x_coordinates, y_coordinates = get_meshgrid(field_of_view, self._grid_size)
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

    def get_source_spectrum(self, source_name: str):
        return self._scene.get_source(source_name)._spectral_energy_distribution

    def get_time_steps(self):
        return self.detector_time_steps

    def get_wavelength_bin_centers(self):
        return self._instrument.wavelength_bin_centers

    def get_wavelength_bin_widths(self):
        return self._instrument.wavelength_bin_widths

    def get_wavelength_bin_edges(self):
        return self._instrument.wavelength_bin_edges

    def set(self, entity: Union[Instrument, Observation, Scene, Configuration]):
        entity._phringe = self
        # entity._device = self._device
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

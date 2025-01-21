from typing import Union

import numpy as np
import torch
from numpy import ndarray
from skimage.measure import block_reduce
from torch import Tensor
from tqdm import tqdm

from phringe.core.entities.configuration import Configuration
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene
from phringe.core.entities.sources.exozodi import Exozodi
from phringe.core.entities.sources.local_zodi import LocalZodi
from phringe.core.entities.sources.planet import Planet
from phringe.core.entities.sources.star import Star
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
            time_step_size: float = 600
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

    def get_counts(self):
        # Move all tensors to the device
        self._instrument.aperture_diameter = self._instrument.aperture_diameter.to(self._device)
        # self._detector_time_steps = self._detector_time_steps.to(self._device)
        # self._instrument.wavelength_bin_centers = self._instrument.wavelength_bin_centers.to(self._device)
        # self._wavelength_bin_widths = self._wavelength_bin_widths.to(self._device)
        # self._wavelength_bin_edges = self._wavelength_bin_edges.to(self._device)
        # self.amplitude_pert_time_series = self.amplitude_pert_time_series.to(self._device)
        # self.phase_pert_time_series = self.phase_pert_time_series.to(self._device)
        # self.polarization_pert_time_series = self.polarization_pert_time_series.to(self._device)
        # self._simulation_time_step_size = self._simulation_time_step_size.to(self._device)
        # self.simulation_time_steps = self.simulation_time_steps.to(self._device)

        # all_sources = self._scene.get_all_sources()

        # for index_source, source in enumerate(all_sources):
        #     self._sources[index_source].spectral_flux_density = source.spectral_flux_density.to(self._device)
        #     self._sources[index_source]._sky_coordinates = source._sky_coordinates.to(self._device)
        #     self._sources[index_source]._sky_brightness_distribution = source._sky_brightness_distribution.to(
        #         self._device)

        # Prepare output tensor
        counts = torch.zeros(
            (self._instrument.number_of_outputs,
             len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            device=self._device
        )

        diff_counts = torch.zeros(
            (len(self._instrument.differential_outputs),
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
                    if not self._detailed and i not in np.array(self._instrument.differential_outputs).flatten():
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
                                *[amplitude_pert_time_series[k][None, it_low:it_high, None, None] for k in
                                  range(self._instrument.number_of_inputs)],
                                *[phase_pert_time_series[k][:, it_low:it_high, None, None] for k in
                                  range(self._instrument.number_of_inputs)],
                                *[torch.tensor(0, device=self._device) for _ in
                                  range(self._instrument.number_of_inputs)],
                                *[polarization_pert_time_series[k][None, it_low:it_high, None, None] for k in
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

        # Calculate differential outputs
        for i in range(len(self._instrument.differential_outputs)):
            diff_counts[i] = counts[self._instrument.differential_outputs[i][0]] - counts[
                self._instrument.differential_outputs[i][1]]

        # # Bin data to from simulation time steps detector time steps
        binning_factor = int(round(len(self.simulation_time_steps) / len(self.detector_time_steps), 0))

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

        return self._data

    def get_diff_counts(self, index):
        pass

    def get_instrument_response(
            self,
            times: Union[float, ndarray, Tensor] = None,
            wavelengths: Union[float, ndarray, Tensor] = None,
            x_coordinates: Union[float, ndarray, Tensor] = None,
            y_coordinates: Union[float, ndarray, Tensor] = None,
            nulling_baseline: float = None,
            output_as_numpy: bool = False,
    ):

        times = self.simulation_time_steps if times is None else times
        wavelengths = self._instrument.wavelength_bin_centers if wavelengths is None else wavelengths
        x_coordinates = self._scene.star._sky_coordinates[0] if x_coordinates is None else x_coordinates
        y_coordinates = self._scene.star._sky_coordinates[1] if y_coordinates is None else y_coordinates
        nulling_baseline = 14

        # An observation object needs to be added to the PHRINGE object before the instrument response can be calculated
        if self._observation is None:
            raise ValueError('An observation is required for the instrument response')

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

        response = torch.stack([self._instrument.response[j](
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
        pass

    def get_wavelength_bin_centers(self):
        return self._instrument.wavelength_bin_centers

    def get_wavelength_bin_widths(self):
        pass

    def get_wavelength_bin_edges(self):
        pass

    def set(self, entity: Union[Instrument, Observation, Scene, Configuration]):
        entity._device = self._device
        if isinstance(entity, Instrument):
            self._instrument = entity
        elif isinstance(entity, Observation):
            self._observation = entity
        elif isinstance(entity, Scene):
            self._scene = entity
        elif isinstance(entity, Configuration):
            self._instrument = Instrument(**entity.dict['instrument'])
            self._observation = Observation(**entity.dict['observation'])
            self._scene = Scene(**entity.dict['scene'])
        else:
            raise ValueError(f'Invalid entity type: {type(entity)}')

        if self._instrument is not None:
            self._instrument._observation = self._observation
            self._instrument._number_of_simulation_time_steps = len(
                self.simulation_time_steps) if self.simulation_time_steps is not None else None

        if self._scene is not None:
            self._scene._device = self._device
            self._scene._instrument = self._instrument
            self._scene._grid_size = self._grid_size
            self._scene._simulation_time_steps = self.simulation_time_steps

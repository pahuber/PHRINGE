from typing import Union

import torch
from numpy import ndarray
from torch import Tensor

from phringe.core.entities.base_entity import BaseEntity
from phringe.core.entities.configuration import Configuration
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene


class PHRINGE:
    """
    Main PHRINGE class.
    """

    def __init__(self, seed: int = None, gpu: int = None, device: torch.device = None, time_step_size: float = 100):
        self._device = self._get_device(gpu) if device is None else device
        self._instrument = None
        self._observation = None
        self._scene = None
        self.seed = seed
        self._time_step_size = time_step_size
        self._simulation_time_steps = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)  # Set the attribute

        # if name == '_instrument' and self._instrument is not None:
        #     self._instrument._phringe = self
        try:
            self._instrument.perturbations.amplitude._modulation_period = value.modulation_period
        except (AttributeError, TypeError):
            pass
        try:
            self._instrument.perturbations.amplitude._number_of_simulation_time_steps = len(self.simulation_time_steps)
        except (AttributeError, TypeError):
            pass
        try:
            self._instrument.perturbations.phase._modulation_period = value.modulation_period
        except (AttributeError, TypeError):
            pass
        try:
            self._instrument.perturbations.phase._number_of_simulation_time_steps = len(self.simulation_time_steps)
        except (AttributeError, TypeError):
            pass
        try:
            self._instrument.perturbations.phase._wavelengths = value.wavelength_bin_centers
        except (AttributeError, TypeError):
            pass
        try:
            self._instrument.perturbations.polarization._modulation_period = value.modulation_period
        except (AttributeError, TypeError):
            pass
        try:
            self._instrument.perturbations.polarization._number_of_simulation_time_steps = len(
                self.simulation_time_steps)
        except (AttributeError, TypeError):
            pass

    @property
    def simulation_time_steps(self):
        return torch.linspace(
            0,
            self._observation.total_integration_time,
            int(self._observation.total_integration_time / self._time_step_size)
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
        pass

    def get_diff_counts(self, index):
        pass

    def get_instrument_response(
            self,
            times: Union[float, ndarray, Tensor],
            wavelengths: Union[float, ndarray, Tensor],
            x_coordinates: Union[float, ndarray, Tensor],
            y_coordinates: Union[float, ndarray, Tensor],
            nulling_baseline: float,
            output_as_numpy: bool = False,
    ):

        # An observation object needs to be added to the PHRINGE object before the instrument response can be calculated
        if self._observation is None:
            raise ValueError('An observation is required for the instrument response')

        # Calculate perturbation time series unless they have been manually set by the user
        amplitude_pert_time_series = self._instrument.perturbations.amplitude.time_series if self._instrument.perturbations.amplitude is not None else torch.zeros(
            (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
            dtype=torch.float32
        )
        phase_pert_time_series = self._instrument.perturbations.phase.time_series if self._instrument.perturbations.phase is not None else torch.zeros(
            (self._instrument.number_of_inputs, len(self._instrument.wavelength_bin_centers),
             len(self.simulation_time_steps)),
            dtype=torch.float32
        )
        polarization_pert_time_series = self._instrument.perturbations.polarization.time_series if self._instrument.perturbations.polarization is not None else torch.zeros(
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

    def get_source_spectrum(self, source_name):
        pass

    def get_time_steps(self):
        pass

    def get_wavelength_bin_centers(self):
        pass

    def get_wavelength_bin_widths(self):
        pass

    def get_wavelength_bin_edges(self):
        pass

    def set(self, entity: BaseEntity):
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

        # Pass along attributes that are required class internally
        # self._instrument._modulation_period = self._observation.modulation_period if self._observation is not None else None
        # self._instrument._number_of_simulation_time_steps = 10000 if self._observation is not None else None  # TODO: Make this a parameter

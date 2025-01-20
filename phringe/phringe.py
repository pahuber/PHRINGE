from typing import Union

import torch
from numpy import ndarray
from phringe.core.event_manager import EventManager
from torch import Tensor

from phringe.core.entities.configuration import Configuration
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene


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
            time_step_size: float = 100
    ):
        self._event_manager = EventManager()
        self._device = self._get_device(gpu) if device is None else device
        self._instrument = None
        self._observation = None
        self._scene = None
        self.seed = seed
        self._grid_size = grid_size
        self._time_step_size = time_step_size
        self._simulation_time_steps = None
        self._detector_time_steps = None

    # def __setattr__(self, name, value):
    #     super().__setattr__(name, value)
    #     # Attributes derived from _observation
    #     # try:
    #     #     # link((self._instrument.perturbations.amplitude, '_modulation_period'),
    #     #     #      (self._observation, 'modulation_period'))
    #     # except (AttributeError, TypeError):
    #     #     pass
    #     # if hasattr(self, '_instrument') and hasattr(self, '_observation'):
    #     #     self._event_manager.register(
    #     #         lambda attr_name, old_val, new_val: self._instrument._get_wavelength_bins(new_val),
    #     #         self._observation, 'modulation_period')
    #
    #     try:
    #         self._instrument.perturbations.amplitude._modulation_period = self._observation.modulation_period
    #     except (AttributeError, TypeError):
    #         pass
    #     try:
    #         self._instrument.perturbations.phase._modulation_period = self._observation.modulation_period
    #     except (AttributeError, TypeError):
    #         pass
    #     try:
    #         self._instrument.perturbations.polarization._modulation_period = self._observation.modulation_period
    #     except (AttributeError, TypeError):
    #         pass
    #
    #     # Attributes derived from _instrument
    #     try:
    #         self._instrument.perturbations.phase._wavelengths = self._instrument.wavelength_bin_centers
    #     except (AttributeError, TypeError):
    #         pass
    #     try:
    #         for source in self._scene.get_all_sources():
    #             # try:
    #             source._wavelength_bin_centers = self._instrument.wavelength_bin_centers
    #             source._grid_size = self._grid_size
    #     except (AttributeError, TypeError):
    #         pass
    #
    #     # Attributes derived from _simulation_time_steps
    #     try:
    #         self._instrument.perturbations.amplitude._number_of_simulation_time_steps = len(
    #             self.simulation_time_steps)
    #     except (AttributeError, TypeError):
    #         pass
    #     try:
    #         self._instrument.perturbations.phase._number_of_simulation_time_steps = len(self.simulation_time_steps)
    #     except (AttributeError, TypeError):
    #         pass
    #     try:
    #         self._instrument.perturbations.polarization._number_of_simulation_time_steps = len(
    #             self.simulation_time_steps)
    #     except (AttributeError, TypeError):
    #         pass

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
            int(self._observation.total_integration_time / self._time_step_size),
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

        for index_source, source in enumerate(self._sources):
            self._sources[index_source].spectral_flux_density = source.spectral_flux_density.to(self._device)
            self._sources[index_source].sky_coordinates = source.sky_coordinates.to(self._device)
            self._sources[index_source].sky_brightness_distribution = source.sky_brightness_distribution.to(
                self._device)

        # Prepare output tensor
        counts = torch.zeros(
            (self._instrument.number_of_outputs,
             len(self._instrument.wavelength_bin_centers),
             len(self._simulation_time_steps)),
            device=self._device
        )

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

        # Calculate perturbation time series unless they have been manually set by the user. If no seed is set, the time
        # series are different every time this method is called
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
        if self._instrument is None:
            raise ValueError('An instrument is required to get the source spectrum')
        if self._scene is None:
            raise ValueError('A scene is required to get the source spectrum')

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
        entity._event_manager = self._event_manager
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

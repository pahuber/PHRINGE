from typing import Any

import numpy as np
from pydantic import BaseModel

from src.sygn.core.entities.base_component import BaseComponent


class Settings(BaseComponent, BaseModel):
    """Class representing the settings."""
    grid_size: int
    has_planet_orbital_motion: bool
    has_stellar_leakage: bool
    has_local_zodi_leakage: bool
    has_exozodi_leakage: bool
    has_amplitude_perturbations: bool
    has_phase_perturbations: bool
    has_polarization_perturbations: bool
    time_steps: Any = None
    wavelength_steps: Any = None

    def _calculate_time_steps(self, observation) -> np.ndarray:
        """Calculate the time steps.

        :param observation: The observation
        :return: The time steps
        """
        # TODO: Implement sensible calculation of the time steps
        number_of_steps = int(observation.total_integration_time / observation.exposure_time)
        return np.linspace(0, observation.total_integration_time, 200)

    def _calculate_wavelength_steps(self, observatory) -> np.ndarray:
        """Calculate the wavelength steps.

        :param observatory: The observatory
        :return: The wavelength steps
        """
        # TODO: Implement sensible calculation of the wavelength steps
        return np.linspace(observatory.wavelength_range_lower_limit, observatory.wavelength_range_upper_limit,
                           100)

    def prepare(self, observation, observatory):
        """Prepare the settings for the simulation.

        :param observation: The observation
        :param observatory: The observatory
        """
        self.time_steps = self._calculate_time_steps(observation)
        self.wavelength_steps = self._calculate_wavelength_steps(observatory)

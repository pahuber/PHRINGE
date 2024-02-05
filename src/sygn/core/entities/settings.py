import numpy as np

from src.sygn.core.entities.base_component import BaseComponent
from pydantic import BaseModel


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

    def _calculate_time_steps(self, observation) -> np.ndarray:
        """Calculate the time steps."""
        number_of_steps = int(observation.total_integration_time / observation.exposure_time)
        return np.linspace(0, observation.total_integration_time, number_of_steps)

    def _calculate_wavelength_steps(self, observatory) -> np.ndarray:
        """Calculate the wavelength steps."""
        # TODO: Implement the calculation of the wavelength steps
        return np.linspace(observatory.wavelength_minimum, observatory.wavelength_maximum, 40)

    def prepare(self, observation, observatory):
        """Prepare the settings for the simulation."""
        self.time_steps = self._calculate_time_steps(observation)
        self.wavelength_steps = self._calculate_wavelength_steps(observatory)

from functools import cached_property
from typing import Any

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from pydantic import BaseModel

from phringe.core.entities.base_component import BaseComponent


class Settings(BaseComponent, BaseModel):
    """Class representing the simulation settings.

    :param grid_size: The size of the grid
    :param has_planet_orbital_motion: Whether the planet has orbital motion
    :param has_stellar_leakage: Whether the stellar leakage is present
    :param has_local_zodi_leakage: Whether the local zodiacal light leakage is present
    :param has_exozodi_leakage: Whether the exozodiacal light leakage is present
    :param has_amplitude_perturbations: Whether amplitude perturbations are present
    :param has_phase_perturbations: Whether phase perturbations are present
    :param has_polarization_perturbations: Whether polarization perturbations are present
    :param simulation_time_steps: The time steps
    :param simulation_wavelength_steps: The wavelength steps
    """
    grid_size: int
    has_planet_orbital_motion: bool
    has_stellar_leakage: bool
    has_local_zodi_leakage: bool
    has_exozodi_leakage: bool
    has_amplitude_perturbations: bool
    has_phase_perturbations: bool
    has_polarization_perturbations: bool
    simulation_time_steps: Any = None
    simulation_wavelength_steps: Any = None
    simulation_wavelength_bin_widths: Any = None

    @cached_property
    def simulation_time_step_duration(self) -> Quantity:
        """Return the simulation time step duration.

        :return: The simulation time step duration
        """
        return (1 * u.min).to(u.s)

    def _calculate_simulation_time_steps(self, observation) -> np.ndarray:
        """Calculate the simulation time steps.

        :param observation: The observation
        :return: The simulation time steps
        """
        number_of_steps = int(observation.total_integration_time.to(u.s) / self.simulation_time_step_duration)
        return np.linspace(0, observation.total_integration_time, number_of_steps)

    def _calculate_simulation_wavelength_bin_widths(self, observatory) -> np.ndarray:
        """Calculate the simulation wavelength bin widths.

        :param observatory: The observatory
        :return: The simulation wavelength bin widths
        """
        current_edge = observatory.wavelength_range_lower_limit
        bin_widths = []
        for index, wavelength in enumerate(self.simulation_wavelength_steps):
            upper_wavelength = self.simulation_wavelength_steps[index + 1] if index < len(
                self.simulation_wavelength_steps) - 1 else observatory.wavelength_range_upper_limit
            bin_widths.append(
                ((wavelength - current_edge) + (upper_wavelength - wavelength) / 2).to(
                    u.um).value)
            current_edge += bin_widths[index] * u.um
        return np.array(bin_widths) * u.um

    def _calculate_simulation_wavelength_steps(self, observatory, scene) -> np.ndarray:
        """Calculate the optimized wavelength sampling for the simulation. This is done by taking the gradient of the
        normalized planet spectra and adding extra wavelength steps (to the instrument wavelength bins) where the
        gradient is larger than 1. This assures a good sampling of the planet spectra if the instrument spectral
        resolving power is low compared to the variation of the spectra.

        :param observatory: The observatory
        :param scene: The scene
        :return: The wavelength steps
        """
        optimized_wavelength_steps = []
        instrument_wavelength_bin_centers = observatory.wavelength_bin_centers
        for planet in scene.planets:
            spectrum_gradient = np.gradient(
                planet.mean_spectral_flux_density.value / np.max(planet.mean_spectral_flux_density.value),
                scene.maximum_simulation_wavelength_steps
            )

            indices = np.where(np.abs(spectrum_gradient) > 1)
            mask = np.zeros(len(scene.maximum_simulation_wavelength_steps))
            mask[indices] = 1

            for index, value in enumerate(mask):
                if value == 1:
                    optimized_wavelength_steps.append(scene.maximum_simulation_wavelength_steps[index])

            optimized_wavelength_steps = optimized_wavelength_steps + list(
                instrument_wavelength_bin_centers.to(u.um).value)
            optimized_wavelength_steps = sorted(optimized_wavelength_steps)

        return np.unique(np.array(optimized_wavelength_steps)) * u.um

    def prepare(self, observation, observatory, scene):
        """Prepare the settings for the simulation.

        :param observation: The observation
        :param observatory: The observatory
        :param scene: The scene
        """
        self.simulation_time_steps = self._calculate_simulation_time_steps(observation)
        self.simulation_wavelength_steps = self._calculate_simulation_wavelength_steps(observatory, scene)
        self.simulation_wavelength_bin_widths = self._calculate_simulation_wavelength_bin_widths(observatory)

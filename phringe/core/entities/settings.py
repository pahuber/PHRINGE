from typing import Any

from pydantic import BaseModel


class Settings(BaseModel):
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

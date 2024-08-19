from typing import Any

import torch
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import tensor

from phringe.io.validators import validate_quantity_units


class Simulation(BaseModel):
    """Class representing the simulation.

    :param grid_size: The size of the grid
    :param time_step_size: The time step size
    :param has_planet_orbital_motion: Whether the planet has orbital motion
    :param has_planet_signal: Whether the planet signal is present
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
    time_step_size: str
    has_planet_orbital_motion: bool
    has_planet_signal: bool
    has_stellar_leakage: bool
    has_local_zodi_leakage: bool
    has_exozodi_leakage: bool
    has_amplitude_perturbations: bool
    has_phase_perturbations: bool
    has_polarization_perturbations: bool
    simulation_time_steps: Any = None
    simulation_wavelength_steps: Any = None
    simulation_wavelength_bin_widths: Any = None

    @field_validator('time_step_size')
    def _validate_time_step_size(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the time step size input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The time step size in units of seconds
        """
        return tensor(
            validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,)).si.value,
            dtype=torch.float32
        )

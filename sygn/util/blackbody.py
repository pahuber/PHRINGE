from typing import Union

import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody
from astropy.units import Quantity


def create_blackbody_spectrum(
        temperature: Quantity,
        wavelength_steps: np.ndarray,
        source_solid_angle: Union[Quantity, np.ndarray]
) -> np.ndarray:
    """Return a blackbody spectrum for an astrophysical object. The spectrum is binned already to the wavelength bin
    centers of the mission.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_steps: Array containing the wavelength steps
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the flux per bin in units of ph m-2 s-1 um-1
    """
    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_steps)

    return _convert_spectrum_units(blackbody_spectrum, wavelength_steps, source_solid_angle)


def _convert_spectrum_units(
        spectrum: np.ndarray,
        wavelength_steps: np.ndarray,
        source_solid_angle: Union[Quantity, np.ndarray]
) -> np.ndarray:
    """Convert the binned black body spectrum from units erg / (Hz s sr cm2) to units ph / (m2 s um)

    :param spectrum: The binned blackbody spectrum
    :param wavelength_steps: The wavelength bin centers
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the spectral flux density in correct units
    """
    spectral_flux_density = np.zeros(len(spectrum)) * u.ph / u.m ** 2 / u.s / u.um

    for index in range(len(spectrum)):
        if source_solid_angle.size == 1:
            solid_angle = source_solid_angle

        # The solid angle can also be an array of values corresponding to the field of view at different wavelengths.
        # This is used e.g. for the local zodi
        else:
            solid_angle = source_solid_angle[index]

        current_spectral_flux_density = (spectrum[index] * (solid_angle).to(u.sr)).to(
            u.ph / u.m ** 2 / u.s / u.um,
            equivalencies=u.spectral_density(
                wavelength_steps[index]))

        spectral_flux_density[index] = current_spectral_flux_density
    return spectral_flux_density

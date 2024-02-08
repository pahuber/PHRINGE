from typing import Union

import astropy.units
import numpy as np
import spectres
from astropy import units as u
from astropy.modeling.models import BlackBody
from astropy.units import Quantity


def create_blackbody_spectrum(temperature: Quantity,
                              wavelength_steps: np.ndarray,
                              source_solid_angle: Union[Quantity, np.ndarray]) -> np.ndarray:
    """Return a blackbody spectrum for an astrophysical object. The spectrum is binned already to the wavelength bin
    centers of the mission.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_steps: Array containing the wavelength steps
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the flux per bin in units of ph m-2 s-1 um-1
    """

    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_steps)

    return _convert_spectrum_units(blackbody_spectrum, wavelength_steps, source_solid_angle)


def create_blackbody_spectrum2(temperature,
                               wavelength_range_lower_limit: Union[astropy.units.Quantity, float],
                               wavelength_range_upper_limit: Union[astropy.units.Quantity, float],
                               wavelength_bin_centers: np.ndarray,
                               wavelength_bin_widths: np.ndarray,
                               source_solid_angle: Union[astropy.units.Quantity, np.ndarray],
                               is_fitting_mode: bool = False,
                               planet_radius: float = None,
                               star_distance: float = None) -> np.ndarray:
    """Return a blackbody spectrum for an astrophysical object. The spectrum is binned already to the wavelength bin
    centers of the mission.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_range_lower_limit: Lower limit of the wavelength range
    :param wavelength_range_upper_limit: Upper limit of the wavelength range
    :param wavelength_bin_centers: Array containing the wavelength bin centers
    :param wavelength_bin_widths: Array containing the wavelength bin widths
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the flux per bin in units of ph m-2 s-1 um-1
    """
    if is_fitting_mode:
        temperature *= u.K
        wavelength_range_lower_limit *= u.um
        wavelength_range_upper_limit *= u.um
        wavelength_bin_centers *= u.um
        wavelength_bin_widths *= u.um
        source_solid_angle = np.pi * (planet_radius / star_distance) ** 2 * u.sr

    wavelength_range = np.linspace(wavelength_range_lower_limit.value, wavelength_range_upper_limit.value + 1,
                                   1000) * wavelength_range_upper_limit.unit
    blackbody_spectrum = BlackBody(temperature=temperature)(wavelength_range)

    units = blackbody_spectrum.unit
    blackbody_spectrum_binned = spectres.spectres(new_wavs=wavelength_bin_centers.to(u.um).value,
                                                  spec_wavs=wavelength_range.to(u.um).value,
                                                  spec_fluxes=blackbody_spectrum.value,
                                                  fill=0) * units

    return _convert_spectrum_units(blackbody_spectrum_binned, wavelength_bin_centers, source_solid_angle,
                                   is_fitting_mode)


def _convert_spectrum_units(spectrum: np.ndarray,
                            wavelength_steps: np.ndarray,
                            source_solid_angle: Union[astropy.units.Quantity, np.ndarray]) -> np.ndarray:
    """Convert the binned black body spectrum from units erg / (Hz s sr cm2) to units ph / (m2 s um)

    :param blackbody_spectrum_binned: The binned blackbody spectrum
    :param wavelength_bin_centers: The wavelength bin centers
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


def _convert_spectrum_units2(blackbody_spectrum_binned: np.ndarray,
                             wavelength_bin_centers: np.ndarray,
                             source_solid_angle: Union[astropy.units.Quantity, np.ndarray],
                             is_fitting_mode: bool = False) -> np.ndarray:
    """Convert the binned black body spectrum from units erg / (Hz s sr cm2) to units ph / (m2 s um)

    :param blackbody_spectrum_binned: The binned blackbody spectrum
    :param wavelength_bin_centers: The wavelength bin centers
    :param source_solid_angle: The solid angle of the source
    :return: Array containing the spectral flux density in correct units
    """
    spectral_flux_density = np.zeros(len(blackbody_spectrum_binned)) * u.ph / u.m ** 2 / u.s / u.um

    for index in range(len(blackbody_spectrum_binned)):
        if source_solid_angle.size == 1:
            solid_angle = source_solid_angle

        # The solid angle can also be an array of values corresponding to the field of view at different wavelengths.
        # This is used e.g. for the local zodi
        else:
            solid_angle = source_solid_angle[index]

        current_spectral_flux_density = (blackbody_spectrum_binned[index] * (solid_angle).to(u.sr)).to(
            u.ph / u.m ** 2 / u.s / u.um,
            equivalencies=u.spectral_density(
                wavelength_bin_centers[index]))

        spectral_flux_density[index] = current_spectral_flux_density
    if is_fitting_mode:
        return spectral_flux_density.value
    return spectral_flux_density

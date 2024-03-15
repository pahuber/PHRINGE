import time

import numpy as np
import torch
from astropy import units as u
from scipy.constants import c, h, k


# def crop_spectral_flux_density_to_wavelength_range(
#         spectral_flux_density: np.ndarray,
#         input_wavelength_range: np.ndarray,
#         wavelength_range_lower_limit: Quantity,
#         wavelength_range_upper_limit: Quantity
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Crop the spectral flux density to the wavelength range of the observatory.
#
#     :param spectral_flux_density: The spectral flux density
#     :param wavelength_range_lower_limit: The lower limit of the wavelength range
#     :param wavelength_range_upper_limit: The upper limit of the wavelength range
#     :return: The cropped spectral flux density and the cropped wavelength range
#     """
#     indices = np.where((input_wavelength_range >= wavelength_range_lower_limit) & (
#             input_wavelength_range <= wavelength_range_upper_limit))
#
#     return spectral_flux_density[indices], input_wavelength_range[indices]
#

def create_blackbody_spectrum(
        temperature: float,
        wavelength_steps: np.ndarray,
) -> np.ndarray:
    """Return a blackbody spectrum for an astrophysical object.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_steps: Array containing the wavelength steps
    :return: Array containing the flux per bin in units of ph m-3 s-1 sr-1
    """

    t0 = time.time_ns()
    b = 2 * h * c ** 2 / wavelength_steps ** 5 / (torch.exp(torch.tensor(
        h * c / (k * wavelength_steps * temperature))) - 1) / c * wavelength_steps / h

    t1 = time.time_ns()

    print(f"Time to create blackbody spectrum: {(t1 - t0) / 1e9} s")

    # blackbody_spectrum = BlackBody(temperature=temperature * u.K)(wavelength_steps.to(u.AA))
    # a = _convert_spectrum_units(blackbody_spectrum, wavelength_steps, source_solid_angle)
    #
    # t2 = time.time_ns()
    #
    # # print(f"Time to create blackbody spectrum: {(t1 - t0) / 1e9} s")
    # print(f"Time to convert blackbody spectrum: {(t2 - t1) / 1e9} s")

    return b  # .value * u.ph / u.s / u.m ** 3


# def convert_spectrum_units(
#         spectrum: np.ndarray,
#         wavelength_steps: np.ndarray,
#         source_solid_angle: Union[Quantity, np.ndarray]
# ) -> np.ndarray:
#     """Convert the binned black body spectrum from units erg / (Hz s sr cm2) to units ph / (m2 s um)
#
#     :param spectrum: The binned blackbody spectrum
#     :param wavelength_steps: The wavelength bin centers
#     :param source_solid_angle: The solid angle of the source
#     :return: Array containing the spectral flux density in correct units
#     """
#     spectral_flux_density = np.zeros(len(spectrum)) * u.ph / u.m ** 2 / u.s / u.um
#
#     solid_angle2 = source_solid_angle.to(u.sr).value
#     spectrum2 = spectrum.value
#     c1 = c
#     h1 = h
#
#     converted = spectrum2 * 1e-11 / h / c * solid_angle2
#
#     for index in range(len(spectrum)):
#         if source_solid_angle.size == 1:
#             solid_angle = source_solid_angle
#
#         # The solid angle can also be an array of values corresponding to the field of view at different wavelengths.
#         # This is used e.g. for the local zodi
#         else:
#             solid_angle = source_solid_angle[index]
#
#         # current_spectral_flux_density =
#
#         current_spectral_flux_density = (spectrum[index] * (solid_angle).to(u.sr)).to(
#             u.ph / u.m ** 3 / u.s,
#             equivalencies=u.spectral_density(
#                 wavelength_steps.to(u.m)[index]))
#
#         spectral_flux_density[index] = current_spectral_flux_density
#     return spectral_flux_density


def convert_spectrum_from_joule_to_photons(
        spectrum: np.ndarray,
        wavelength_steps: np.ndarray,
) -> np.ndarray:
    """Convert the binned black body spectrum from units W / (sr m3) to units ph / (m3 s sr)

    :param spectrum: The binned blackbody spectrum
    :param wavelength_steps: The wavelength bin centers
    :return: Array containing the spectral flux density in correct units
    """
    spectral_flux_density = np.zeros(len(spectrum)) * u.ph / u.m ** 3 / u.s / u.sr

    # spectrum2 = spectrum.value
    # c1 = c
    # h1 = h
    #
    # converted = spectrum2 * 1e-11 / h / c

    for index in range(len(spectrum)):
        # current_spectral_flux_density =

        current_spectral_flux_density = (spectrum[index]).to(
            u.ph / u.m ** 3 / u.s / u.sr,
            equivalencies=u.spectral_density(
                wavelength_steps.to(u.m)[index]))

        spectral_flux_density[index] = current_spectral_flux_density
    return spectral_flux_density

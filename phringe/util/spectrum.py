from pathlib import Path

import numpy as np
import spectres
import torch
from astropy import units as u
from scipy.constants import c, h, k
from torch import Tensor


class InputSpectrum:

    def __init__(
            self,
            path_to_spectrum: Path = None,
            fluxes: Tensor = None,
            wavelengths: Tensor = None,
    ):
        self.path_to_spectrum = path_to_spectrum
        self.fluxes = fluxes
        self.wavelengths = wavelengths

    def get_spectral_energy_distribution(
            self,
            wavelength_bin_centers: Tensor,
            solid_angle: Tensor,
            device: torch.device
    ) -> Tensor:
        """Return the spectrum from a file."""
        # fluxes, wavelengths = TXTReader.read(path_to_spectrum)
        if self.path_to_spectrum is not None:
            spectrum = np.loadtxt(self.path_to_spectrum, usecols=(0, 1))
            fluxes = spectrum[:, 1] * 1e6 * u.W / u.sr / u.m ** 3
            wavelengths = spectrum[:, 0] * 1e-6 * u.m
            fluxes = convert_spectrum_from_joule_to_photons(fluxes, wavelengths)
            fluxes = fluxes.value
            wavelengths = wavelengths.value
        else:
            fluxes = self.fluxes
            wavelengths = self.wavelengths

        binned_spectral_flux_density = spectres.spectres(
            wavelength_bin_centers.cpu().numpy(),
            wavelengths,
            fluxes,
            fill=0,
            verbose=False
        ) * solid_angle
        return torch.asarray(binned_spectral_flux_density, dtype=torch.float32, device=device)


def create_blackbody_spectrum(
        temperature: float,
        wavelength_steps: Tensor
) -> Tensor:
    """Return a blackbody spectrum for an astrophysical object.

    :param temperature: Temperature of the astrophysical object
    :param wavelength_steps: Array containing the wavelength steps
    :return: Array containing the flux per bin in units of ph m-3 s-1 sr-1
    """
    return 2 * h * c ** 2 / wavelength_steps ** 5 / (
            torch.exp(torch.asarray(h * c / (k * wavelength_steps * temperature))) - 1) / c * wavelength_steps / h


def convert_spectrum_from_joule_to_photons(
        spectrum: Tensor,
        wavelength_steps: Tensor,
) -> Tensor:
    """Convert the binned black body spectrum from units W / (sr m3) to units ph / (m3 s sr)

    :param spectrum: The binned blackbody spectrum
    :param wavelength_steps: The wavelength bin centers
    :return: Array containing the spectral flux density in correct units
    """
    spectral_flux_density = np.zeros(len(spectrum)) * u.ph / u.m ** 3 / u.s / u.sr

    for index in range(len(spectrum)):
        # current_spectral_flux_density =

        current_spectral_flux_density = (spectrum[index]).to(
            u.ph / u.m ** 3 / u.s / u.sr,
            equivalencies=u.spectral_density(
                wavelength_steps.to(u.m)[index]))

        spectral_flux_density[index] = current_spectral_flux_density
    return spectral_flux_density

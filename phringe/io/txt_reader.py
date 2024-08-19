from pathlib import Path

import numpy as np
import torch
from astropy import units as u

from phringe.util.spectrum import convert_spectrum_from_joule_to_photons


class TXTReader:
    """Class representation of a text file reader.
    """

    @staticmethod
    def read(file_path: Path) -> (np.ndarray):
        """Read a text file containing a spectrum and return the fluxes and wavelengths.

        :param file_path: The path to the text file
        :return: The fluxes in units of ph/m3/s/sr and wavelengths in untis of m
        """
        spectrum = np.loadtxt(file_path, usecols=(0, 1))
        fluxes = spectrum[:, 1] * 1e6 * u.W / u.sr / u.m ** 3
        wavelengths = spectrum[:, 0] * 1e-6 * u.m
        fluxes = convert_spectrum_from_joule_to_photons(fluxes, wavelengths)

        return torch.asarray(fluxes.value, dtype=torch.float32), torch.asarray(wavelengths.value, dtype=torch.float32)

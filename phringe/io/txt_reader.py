from pathlib import Path
from astropy import units as u
import numpy as np


class TXTReader:
    """Class representation of a text file reader.
    """

    def read(self, file_path: Path) -> (np.ndarray):
        """Read a text file containing a spectrum and return the fluxes and wavelengths.

        :param file_path: The path to the text file
        :return: The fluxes and wavelengths
        """
        spectrum = np.loadtxt(file_path, usecols=(0, 1))
        fluxes = spectrum[:, 1] * u.W / u.sr / u.m ** 2 / u.um
        wavelengths = spectrum[:, 0] * u.um
        return fluxes, wavelengths

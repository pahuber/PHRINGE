import torch
from scipy.constants import c, h, k
from torch import Tensor


def get_blackbody_spectrum_standard_units(
        temperature: float,
        wavelengths: Tensor
) -> Tensor:
    """Return a blackbody spectrum for an astrophysical object.

    :param temperature: Temperature of the astrophysical object
    :param wavelengths: Array containing the wavelength steps
    :return: Array containing the flux per bin in units of ph m-3 s-1 sr-1
    """
    return 2 * h * c ** 2 / wavelengths ** 5 / (
            torch.exp(torch.asarray(h * c / (k * wavelengths * temperature))) - 1) / c * wavelengths / h

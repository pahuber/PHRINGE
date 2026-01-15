import unittest

import astropy.units as u
import numpy as np
import torch

from phringe.io.input_spectrum import InputSpectrum


class TestInputSpectrum(unittest.TestCase):

    def setUp(self):
        self.planet_radius = 1 * u.Rearth
        self.star_distance = 10 * u.pc

        self.wavelengths = torch.tensor([4, 11, 18], dtype=torch.float32)
        self.frequencies = torch.tensor([1.7e4, 2.7e4, 7.5e4], dtype=torch.float32)

        self.sed_ph_s_m3_sr = torch.tensor([1, 2, 3], dtype=torch.float32)
        self.sed_ph_s_m3 = self.sed_ph_s_m3_sr * np.pi * (self.planet_radius / self.star_distance) ** 2
        ß

        # Common setup for tests
        # self.path_to_spectrum = Path("test_spectrum.txt")
        self.fluxes = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.wavelengths = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
        self.wavelength_bin_centers = torch.tensor([0.6, 1.2], dtype=torch.float32)
        self.solid_angle = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.device = torch.device("cpu")

    def test_initialization_with_data(self):
        spectrum = InputSpectrum(fluxes=self.fluxes, wavelengths=self.wavelengths)
        self.assertIsNone(spectrum.path_to_file)
        self.assertTrue(torch.equal(spectrum.flux, self.fluxes))
        self.assertTrue(torch.equal(spectrum.wavelengths, self.wavelengths))

    def test_read_txt_file(self):
        # This test assumes the existence of a test file "test_spectrum.txt"
        # with known content for validation.
        spectrum = InputSpectrum(path_to_file="test_spectrum.txt")
        spectrum._read_txt_file()
        self.assertIsNotNone(spectrum.sed)
        self.assertIsNotNone(spectrum.wavelengths)


if __name__ == "__main__":
    unittest.main()

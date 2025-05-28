import unittest

import torch

from phringe.util.spectrum import InputSpectrum


class TestInputSpectrum(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        # self.path_to_spectrum = Path("test_spectrum.txt")
        self.fluxes = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.wavelengths = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
        self.wavelength_bin_centers = torch.tensor([0.6, 1.2], dtype=torch.float32)
        self.solid_angle = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.device = torch.device("cpu")

    def test_initialization_with_data(self):
        spectrum = InputSpectrum(fluxes=self.fluxes, wavelengths=self.wavelengths)
        self.assertIsNone(spectrum.path_to_spectrum)
        self.assertTrue(torch.equal(spectrum.fluxes, self.fluxes))
        self.assertTrue(torch.equal(spectrum.wavelengths, self.wavelengths))


if __name__ == "__main__":
    unittest.main()

from typing import Union

import numpy as np
import torch
from scipy.constants import sigma
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_si_units


class Exozodi(BaseSource):
    """Class representation of an exozodi.

    Parameters
    ----------
    level : float
        The level of the exozodi in local zodi levels.
    """
    level: float

    @property
    def _radial_fov_au(self) -> Tensor:
        """Return the radial field of view in AU as a tensor of shape 2 x n_wavelengths x n_grid x n_grid.

        Returns
        -------
        torch.Tensor
            The radial field of view in AU.
        """
        meter_to_au = 6.68459e-12
        host_star_distance = (
            self._phringe._scene.star.distance
            if self._phringe._scene.star is not None
            else self._phringe._observation.host_star_distance
        )
        fov_au = self._phringe._instrument._field_of_view * host_star_distance * meter_to_au

        sky_coordinates = get_meshgrid(fov_au, self._phringe._grid_size, self._phringe._device, )
        radial_fov_map = torch.sqrt(sky_coordinates[0] ** 2 + sky_coordinates[1] ** 2)

        return radial_fov_map

    @property
    def n_grid_points(self) -> int:
        return self._phringe._grid_size ** 2

    @property
    def sky_brightness_distribution(self) -> Tensor:
        device = self._phringe._device

        if self._phringe._scene.star is not None:
            host_star_luminosity = self._phringe._scene.star.luminosity
        else:
            host_star_luminosity = 4 * np.pi * self._phringe._observation.host_star_radius ** 2 * sigma * self._phringe._observation.host_star_temperature ** 4

        ref_radius_au = torch.sqrt(torch.tensor(host_star_luminosity / 3.86e26, device=device, dtype=torch.float32))
        surface_maps = self.level * 7.12e-8 * (self._radial_fov_au / ref_radius_au) ** (-0.34)

        sky_brightness_distribution = surface_maps * self.spectral_energy_distribution

        # Broadcast to time dimension
        return sky_brightness_distribution[:, None, :, :]

    @property
    def sky_coordinates(self) -> Tensor:
        sky_coordinates = get_meshgrid(
            self._phringe._instrument._field_of_view,
            self._phringe._grid_size,
            self._phringe._device,
        )

        # Broadcast to time dimension
        return sky_coordinates[:, :, None, :, :]

    @property
    def solid_angle(self) -> Union[float, Tensor]:
        return self._phringe._instrument._field_of_view ** 2

    @property
    def spectral_energy_distribution(self) -> Tensor:
        if self._phringe._scene.star is not None:
            host_star_luminosity = self._phringe._scene.star.luminosity
        else:
            host_star_luminosity = 4 * np.pi * self._phringe._observation.host_star_radius ** 2 * sigma * self._phringe._observation.host_star_temperature ** 4

        # As described by LIFE II (Dannert+2022)
        temperature_map = (278.3 * (host_star_luminosity / 3.86e26) ** 0.25 * self._radial_fov_au ** (-0.5))

        spectral_energy_distribution = (
                get_blackbody_spectrum_si_units(
                    temperature_map,
                    self._phringe._instrument.wavelength_bin_centers[:, None, None]
                )
                * self.solid_angle[:, None, None]
        )

        return spectral_energy_distribution

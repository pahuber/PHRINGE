from typing import Any

import astropy.units as u
import numpy as np
import torch
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.core.sources.base_source import BaseSource
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_radial_map, get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_standard_units


class Exozodi(BaseSource):
    """Class representation of an exozodi.

    Parameters
    ----------
    level : float
        The level of the exozodi in local zodi levels.
    host_star_luminosity : float, optional
        The luminosity of the host star in units of luminosity. Only required if no host star is specified in the scene.
    host_star_distance : float, optional
        The distance to the host star in units of length. Only required if no host star is specified in the scene.
    """
    level: float
    host_star_luminosity: Any = None
    host_star_distance: Any = None

    @field_validator('host_star_luminosity')
    def _validate_host_star_luminosity(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the host star luminosity input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The host star luminosity in units of luminosity
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.W,))

    @field_validator('host_star_distance')
    def _validate_host_star_distance(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the host star distance input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The host star distance in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @property
    def _field_of_view_in_au_radial_map(self):
        host_star_distance = self.host_star_distance if self.host_star_distance is not None else self._phringe._scene.star.distance
        field_of_view_in_au = self._phringe._instrument._field_of_view * host_star_distance * 6.68459e-12
        num_wavelengths = len(self._phringe._instrument._field_of_view)
        shape = (num_wavelengths, self._phringe._grid_size, self._phringe._grid_size)

        field_of_view_in_au_radial_map = torch.zeros(shape, dtype=torch.float32, device=self._phringe._device)

        for index_fov, fov_in_au in enumerate(field_of_view_in_au):
            field_of_view_in_au_radial_map[index_fov] = get_radial_map(fov_in_au, self._phringe._grid_size,
                                                                       self._phringe._device)

        return field_of_view_in_au_radial_map

    @property
    def _sky_brightness_distribution(self):
        host_star_luminosity = self.host_star_luminosity if self.host_star_luminosity is not None else self._phringe._scene.star.luminosity
        reference_radius_in_au = torch.sqrt(
            torch.tensor(host_star_luminosity / 3.86e26, device=self._phringe._device, dtype=torch.float32))
        surface_maps = self.level * 7.12e-8 * (self._field_of_view_in_au_radial_map / reference_radius_in_au) ** (-0.34)
        return surface_maps * self._spectral_energy_distribution

    @property
    def _sky_coordinates(self):
        sky_coordinates = torch.zeros(
            (2, len(self._phringe._instrument._field_of_view), self._phringe._grid_size, self._phringe._grid_size),
            dtype=torch.float32,
            device=self._phringe._device
        )

        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(len(self._phringe._instrument._field_of_view)):
            sky_coordinates_at_fov = get_meshgrid(
                self._phringe._instrument._field_of_view[index_fov],
                self._phringe._grid_size,
                self._phringe._device
            )
            sky_coordinates[:, index_fov] = torch.stack((sky_coordinates_at_fov[0], sky_coordinates_at_fov[1]))

        return sky_coordinates

    @property
    def _solid_angle(self) -> np.ndarray:
        return self._phringe._instrument._field_of_view ** 2

    @property
    def _spectral_energy_distribution(self):
        host_star_luminosity = self.host_star_luminosity if self.host_star_luminosity is not None else self._phringe._scene.star.luminosity
        temperature_map = self._get_temperature_profile(
            self._field_of_view_in_au_radial_map,
            host_star_luminosity
        )

        num_wavelengths = len(self._phringe._instrument._field_of_view)
        shape = (num_wavelengths, self._phringe._grid_size, self._phringe._grid_size)
        spectral_energy_distribution = torch.zeros(shape, dtype=torch.float32, device=self._phringe._device)

        for ifov, fov in enumerate(self._phringe._instrument._field_of_view):
            spectral_energy_distribution[ifov] = get_blackbody_spectrum_standard_units(
                temperature_map[ifov, :, :],
                self._phringe._instrument.wavelength_bin_centers[ifov, None, None]
            ) * self._solid_angle[ifov, None, None]

        return spectral_energy_distribution

    def _get_temperature_profile(
            self,
            maximum_stellar_separations_radial_map: np.ndarray,
            star_luminosity: Quantity
    ) -> np.ndarray:
        """Return a 2D map corresponding to the temperature distribution of the exozodi.

        :param maximum_stellar_separations_radial_map: The 2D map corresponding to the maximum radial stellar
        separations
        :param star_luminosity: The luminosity of the star
        :return: The temperature distribution map
        """
        return (278.3 * (star_luminosity / 3.86e26) ** 0.25 * maximum_stellar_separations_radial_map ** (
            -0.5))

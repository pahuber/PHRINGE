import warnings
from typing import Any

import numpy as np
import torch
from astropy.units import Quantity

from phringe.entities.sources.base_source import BaseSource
from phringe.util.grid import get_radial_map, get_meshgrid
from phringe.util.helpers import Coordinates
from phringe.util.spectrum import create_blackbody_spectrum
from phringe.util.warning import MissingRequirementWarning


class Exozodi(BaseSource):
    """Class representation of an exozodi.
    """
    name: str = 'Exozodi'
    level: float
    # inclination: Any
    field_of_view_in_au_radial_maps: Any = None
    _host_star_luminosity: Any = None
    _host_star_distance: Any = None
    _field_of_view: Any = None

    @property
    def _sky_brightness_distribution(self):
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_brightness_distribution'))
            return None

        if self._grid_size is None:
            warnings.warn(MissingRequirementWarning('Grid size', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='sky_brightness_distribution',
            compute_func=self._get_sky_brightness_distribution,
            required_attributes=(
                self._instrument,
                self._grid_size,
                self._host_star_luminosity,
                self.field_of_view_in_au_radial_maps
            )
        )

    def _sky_brightness_distribution(self) -> np.ndarray:
        reference_radius_in_au = torch.sqrt(torch.tensor(self._host_star_luminosity))
        surface_maps = self.level * 7.12e-8 * (self.field_of_view_in_au_radial_maps / reference_radius_in_au) ** (-0.34)
        return surface_maps * self.spectral_flux_density

    @property
    def _sky_coordinates(self):
        if self._grid_size is None:
            warnings.warn(MissingRequirementWarning('Grid size', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='sky_coordinates',
            compute_func=self._get_sky_coordinates,
            required_attributes=(
                self._grid_size,
                self._field_of_view
            )
        )

    def _get_sky_coordinates(self) -> Coordinates:
        sky_coordinates = torch.zeros((2, len(self._field_of_view), self._grid_size, self._grid_size),
                                      dtype=torch.float32, device=self._device)

        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(len(self._field_of_view)):
            sky_coordinates_at_fov = get_meshgrid(
                self._field_of_view[index_fov],
                self._grid_size)
            sky_coordinates[:, index_fov] = torch.stack(
                (sky_coordinates_at_fov[0], sky_coordinates_at_fov[1]))

        return sky_coordinates

    @property
    def _solid_angle(self) -> np.ndarray:
        return self._get_cached_value(
            attribute_name='solid_angle',
            compute_func=self._get_solid_angle,
            required_attributes=(
                (self._field_of_view,)
            )
        )

    def _get_solid_angle(self) -> float:
        """Calculate and return the solid angle of the exozodi.

        :param kwargs: Additional keyword arguments
        :return: The solid angle
        """
        return self._field_of_view ** 2

    @property
    def _spectral_energy_distribution(self):

        return self._get_cached_value(
            attribute_name='spectral_energy_distribution',
            compute_func=self._get_spectral_energy_distribution,
            required_attributes=(
                self._field_of_view,
                self._host_star_distance,
                self._host_star_luminosity,
                self._grid_size
            )
        )

    def _get_spectral_energy_distribution(self) -> np.ndarray:
        field_of_view = kwargs['field_of_view']
        star_distance = kwargs['star_distance']
        star_luminosity = kwargs['star_luminosity']
        field_of_view_in_au = field_of_view * star_distance * 6.68459e-12
        num_wavelengths = len(field_of_view)
        shape = (num_wavelengths, grid_size, grid_size)

        self.field_of_view_in_au_radial_maps = torch.zeros(shape, dtype=torch.float32)
        spectral_flux_density = torch.zeros(shape, dtype=torch.float32)

        for index_fov, fov_in_au in enumerate(field_of_view_in_au):
            self.field_of_view_in_au_radial_maps[index_fov] = get_radial_map(fov_in_au, grid_size)

        temperature_map = self._get_temperature_profile(
            self.field_of_view_in_au_radial_maps,
            star_luminosity
        )

        for ifov, fov in enumerate(field_of_view):
            spectral_flux_density[ifov] = create_blackbody_spectrum(
                temperature_map[ifov, :, :],
                wavelength_steps[ifov, None, None]
            ) * self._get_solid_angle[ifov, None, None]

        return spectral_flux_density

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
        return (278.3 * star_luminosity ** 0.25 * maximum_stellar_separations_radial_map ** (
            -0.5))

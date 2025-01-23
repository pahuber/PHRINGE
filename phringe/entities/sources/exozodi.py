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
    level: float
    # inclination: Any
    # field_of_view_in_au_radial_maps: Any = None
    host_star_luminosity: Any = None
    host_star_distance: Any = None

    @property
    def _sky_brightness_distribution(self):
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_brightness_distribution'))
            return None

        if self.host_star_luminosity is None:
            warnings.warn(MissingRequirementWarning('Star', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='sky_brightness_distribution',
            compute_func=self._get_sky_brightness_distribution,
            required_attributes=(
                self._grid_size,
                self.host_star_luminosity,
                self._field_of_view_in_au_radial_map
            )
        )

    def _get_sky_brightness_distribution(self) -> np.ndarray:
        reference_radius_in_au = torch.sqrt(torch.tensor(self.host_star_luminosity))
        surface_maps = self.level * 7.12e-8 * (self._field_of_view_in_au_radial_map / reference_radius_in_au) ** (-0.34)
        return surface_maps * self._spectral_energy_distribution

    @property
    def _sky_coordinates(self):
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_coordinates'))
            return None

        return self._get_cached_value(
            attribute_name='sky_coordinates',
            compute_func=self._get_sky_coordinates,
            required_attributes=(
                self._grid_size,
                self._instrument._field_of_view
            )
        )

    def _get_sky_coordinates(self) -> Coordinates:
        sky_coordinates = torch.zeros((2, len(self._instrument._field_of_view), self._grid_size, self._grid_size),
                                      dtype=torch.float32, device=self._device)

        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(len(self._instrument._field_of_view)):
            sky_coordinates_at_fov = get_meshgrid(
                self._instrument._field_of_view[index_fov],
                self._grid_size,
                self._device)
            sky_coordinates[:, index_fov] = torch.stack(
                (sky_coordinates_at_fov[0], sky_coordinates_at_fov[1]))

        return sky_coordinates

    @property
    def _solid_angle(self) -> np.ndarray:
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'solid_angle'))
            return None

        return self._get_cached_value(
            attribute_name='solid_angle',
            compute_func=self._get_solid_angle,
            required_attributes=(
                (self._instrument._field_of_view,)
            )
        )

    def _get_solid_angle(self) -> float:
        """Calculate and return the solid angle of the exozodi.

        :param kwargs: Additional keyword arguments
        :return: The solid angle
        """
        return self._instrument._field_of_view ** 2

    @property
    def _spectral_energy_distribution(self):

        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'spectral_energy_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='spectral_energy_distribution',
            compute_func=self._get_spectral_energy_distribution,
            required_attributes=(
                self._instrument._field_of_view,
                self._instrument.wavelength_bin_centers,
                self.host_star_distance,
                self.host_star_luminosity,
                self._grid_size
            )
        )

    @property
    def _field_of_view_in_au_radial_map(self):
        return self._get_cached_value(
            attribute_name='field_of_view_in_au_radial_map',
            compute_func=self._get_field_of_view_in_au_radial_map,
            required_attributes=(
                self._instrument._field_of_view,
                self._instrument.wavelength_bin_centers,
                self.host_star_distance,
                self._grid_size
            )
        )

    def _get_field_of_view_in_au_radial_map(self):
        field_of_view_in_au = self._instrument._field_of_view * self.host_star_distance * 6.68459e-12
        num_wavelengths = len(self._instrument._field_of_view)
        shape = (num_wavelengths, self._grid_size, self._grid_size)

        field_of_view_in_au_radial_map = torch.zeros(shape, dtype=torch.float32, device=self._device)

        for index_fov, fov_in_au in enumerate(field_of_view_in_au):
            field_of_view_in_au_radial_map[index_fov] = get_radial_map(fov_in_au, self._grid_size, self._device)

        return field_of_view_in_au_radial_map

    def _get_spectral_energy_distribution(self) -> np.ndarray:

        temperature_map = self._get_temperature_profile(
            self._field_of_view_in_au_radial_map,
            self.host_star_luminosity
        )

        num_wavelengths = len(self._instrument._field_of_view)
        shape = (num_wavelengths, self._grid_size, self._grid_size)
        spectral_energy_distribution = torch.zeros(shape, dtype=torch.float32, device=self._device)

        for ifov, fov in enumerate(self._instrument._field_of_view):
            spectral_energy_distribution[ifov] = create_blackbody_spectrum(
                temperature_map[ifov, :, :],
                self._instrument.wavelength_bin_centers[ifov, None, None]
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
        return (278.3 * star_luminosity ** 0.25 * maximum_stellar_separations_radial_map ** (
            -0.5))

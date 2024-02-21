from typing import Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from pydantic import BaseModel

from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.util.blackbody import create_blackbody_spectrum
from phringe.util.grid import get_meshgrid
from phringe.util.helpers import Coordinates


class LocalZodi(BasePhotonSource, BaseModel):
    """Class representation of a local zodi."""

    def _calculate_mean_spectral_flux_density(
            self,
            wavelength_steps: np.ndarray,
            grid_size: int,
            **kwargs
    ) -> np.ndarray:
        """Calculate the mean spectral flux density of the local zodi as described in Dannert et al. 2022.

        :param wavelength_steps: The wavelength steps
        """
        field_of_view = kwargs['field_of_view']
        star_right_ascension = kwargs['star_right_ascension']
        star_declination = kwargs['star_declination']
        variable_tau = 4e-8
        variable_a = 0.22
        ecliptic_latitude, relative_ecliptic_longitude = self._get_ecliptic_coordinates(star_right_ascension,
                                                                                        star_declination)
        mean_spectral_flux_density = (
                variable_tau
                * (create_blackbody_spectrum(265 * u.K, wavelength_steps, field_of_view ** 2)
                   + variable_a * create_blackbody_spectrum(5778 * u.K, wavelength_steps, field_of_view ** 2)
                   * ((1 * u.Rsun).to(u.au) / (1.5 * u.au)) ** 2)
                * (
                        (np.pi / np.arccos(
                            np.cos(relative_ecliptic_longitude) * np.cos(ecliptic_latitude)).value) / (
                                np.sin(ecliptic_latitude) ** 2 + 0.6 * (
                                wavelength_steps / (
                                11 * u.um)) ** (
                                    -0.4) * np.cos(ecliptic_latitude) ** 2)) ** 0.5)
        return mean_spectral_flux_density

    def _calculate_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        grid = np.ones((grid_size, grid_size))
        return np.einsum('i, jk ->ijk', self.mean_spectral_flux_density, grid)

    def _calculate_sky_coordinates(self, grid_size, **kwargs) -> Coordinates:
        number_of_wavelength_steps = kwargs['number_of_wavelength_steps']
        field_of_view = kwargs['field_of_view']

        sky_coordinates = np.zeros(number_of_wavelength_steps, dtype=object)
        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(number_of_wavelength_steps):
            sky_coordinates_at_fov = get_meshgrid(field_of_view[index_fov].to(u.rad), grid_size)
            sky_coordinates[index_fov] = Coordinates(sky_coordinates_at_fov[0], sky_coordinates_at_fov[1])
        return sky_coordinates

    def _get_ecliptic_coordinates(self, star_right_ascension, star_declination) -> Tuple:
        """Return the ecliptic latitude and relative ecliptic longitude that correspond to the star position in the sky.

        :param star_right_ascension: The right ascension of the star
        :param star_declination: The declination of the star
        :return: Tuple containing the two coordinates
        """
        coordinates = SkyCoord(ra=star_right_ascension, dec=star_declination, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.deg)
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.deg)
        # TODO: Fix relative ecliptic longitude with current sun position
        relative_ecliptic_longitude = ecliptic_longitude - 0 * u.deg
        return ecliptic_latitude, relative_ecliptic_longitude

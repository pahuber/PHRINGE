from typing import Tuple

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from pydantic import BaseModel

from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.util.grid import get_meshgrid
from phringe.util.helpers import Coordinates
from phringe.util.spectrum import create_blackbody_spectrum


class LocalZodi(BasePhotonSource, BaseModel):
    """Class representation of a local zodi."""
    name: str = 'LocalZodi'

    def _get_ecliptic_coordinates(self, star_right_ascension, star_declination, solar_ecliptic_latitude) -> Tuple:
        """Return the ecliptic latitude and relative ecliptic longitude that correspond to the star position in the sky.

        :param star_right_ascension: The right ascension of the star
        :param star_declination: The declination of the star
        :param solar_ecliptic_latitude: The ecliptic latitude of the sun
        :return: Tuple containing the two coordinates
        """
        coordinates = SkyCoord(ra=star_right_ascension, dec=star_declination, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.rad).value
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.rad).value
        relative_ecliptic_longitude = ecliptic_longitude - solar_ecliptic_latitude
        return ecliptic_latitude, relative_ecliptic_longitude

    def _get_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        grid = torch.ones((grid_size, grid_size), dtype=torch.float32)
        return torch.einsum('i, jk ->ijk', self.spectral_flux_density, grid)

    def _get_sky_coordinates(self, grid_size, **kwargs) -> Coordinates:
        number_of_wavelength_steps = kwargs['number_of_wavelength_steps']
        field_of_view = kwargs['field_of_view']

        sky_coordinates = torch.zeros((2, number_of_wavelength_steps, grid_size, grid_size))
        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(number_of_wavelength_steps):
            sky_coordinates_at_fov = get_meshgrid(field_of_view[index_fov], grid_size)
            sky_coordinates[:, index_fov] = torch.stack(
                (sky_coordinates_at_fov[0], sky_coordinates_at_fov[1]))
        return sky_coordinates

    def _get_solid_angle(self, **kwargs) -> float:
        """Calculate and return the solid angle of the local zodi.

        :param kwargs: Additional keyword arguments
        :return: The solid angle
        """
        return kwargs['field_of_view'] ** 2

    def _get_spectral_flux_density(
            self,
            wavelength_steps: np.ndarray,
            grid_size: int,
            **kwargs
    ) -> np.ndarray:
        """Calculate the mean spectral flux density of the local zodi as described in Dannert et al. 2022.

        :param wavelength_steps: The wavelength steps
        """
        star_right_ascension = kwargs['star_right_ascension']
        star_declination = kwargs['star_declination']
        solar_ecliptic_latitude = kwargs['solar_ecliptic_latitude']
        variable_tau = 4e-8
        variable_a = 0.22
        ecliptic_latitude, relative_ecliptic_longitude = self._get_ecliptic_coordinates(
            star_right_ascension,
            star_declination,
            solar_ecliptic_latitude
        )
        spectral_flux_density = (
                variable_tau *
                (
                        create_blackbody_spectrum(265, wavelength_steps) * self.solid_angle
                        + variable_a
                        * create_blackbody_spectrum(5778, wavelength_steps) * self.solid_angle
                        * ((1 * u.Rsun).to(u.au) / (1.5 * u.au)).value ** 2
                ) *
                ((torch.pi / torch.arccos(torch.cos(torch.tensor(relative_ecliptic_longitude)) * torch.cos(
                    torch.tensor(ecliptic_latitude))))
                 / (torch.sin(torch.tensor(ecliptic_latitude)) ** 2 + 0.6 * (wavelength_steps / (11e-6)) ** (
                            -0.4) * torch.cos(
                            torch.tensor(ecliptic_latitude)) ** 2)) ** 0.5)
        return spectral_flux_density

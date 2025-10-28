from typing import Tuple, Union, Any

import torch
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.core.sources.base_source import BaseSource
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_standard_units


class LocalZodi(BaseSource):
    """Class representation of a local zodi.

    Parameters
    ----------
    host_star_right_ascension : str, float, or Quantity, optional
        The right ascension of the host star in units of degrees. Only required if not host star is specified in the scene.
    host_star_declination : str, float, or Quantity, optional
        The declination of the host star in units of degrees. Only required if not host star is specified in the scene.
    """
    host_star_right_ascension: Union[str, float, Quantity] = None
    host_star_declination: Union[str, float, Quantity] = None
    _solar_ecliptic_latitude: Union[str, float, Quantity] = None

    @field_validator('host_star_right_ascension')
    def _validate_right_ascension(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the right ascension input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The right ascension in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('host_star_declination')
    def _validate_declination(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the declination input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The declination in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    def _get_ecliptic_coordinates(self, star_right_ascension, star_declination, solar_ecliptic_latitude) -> Tuple:
        """Return the ecliptic latitude and relative ecliptic longitude that correspond to the star position in the sky.

        :param star_right_ascension: The right ascension of the star
        :param star_declination: The declination of the star
        :param solar_ecliptic_latitude: The ecliptic latitude of the sun
        :return: Tuple containing the two coordinates
        """
        coordinates = SkyCoord(ra=star_right_ascension * u.rad, dec=star_declination * u.rad, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.rad).value
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.rad).value
        relative_ecliptic_longitude = ecliptic_longitude - solar_ecliptic_latitude
        return ecliptic_latitude, relative_ecliptic_longitude

    @property
    def _sky_brightness_distribution(self):
        grid = torch.ones((self._phringe._grid_size, self._phringe._grid_size), dtype=torch.float32,
                          device=self._phringe._device)
        return torch.einsum('i, jk ->ijk', self._spectral_energy_distribution, grid)

    @property
    def _sky_coordinates(self):
        number_of_wavelength_steps = len(self._phringe._instrument.wavelength_bin_centers)

        sky_coordinates = torch.zeros(
            (2, number_of_wavelength_steps, self._phringe._grid_size, self._phringe._grid_size),
            device=self._phringe._device)
        # The sky coordinates have a different extent for each field of view, i.e. for each wavelength
        for index_fov in range(number_of_wavelength_steps):
            sky_coordinates_at_fov = get_meshgrid(self._phringe._instrument._field_of_view[index_fov],
                                                  self._phringe._grid_size,
                                                  device=self._phringe._device)
            sky_coordinates[:, index_fov] = torch.stack(
                (sky_coordinates_at_fov[0], sky_coordinates_at_fov[1]))
        return sky_coordinates

    @property
    def _solid_angle(self):
        return self._phringe._instrument._field_of_view ** 2

    @property
    def _spectral_energy_distribution(self) -> torch.Tensor:
        variable_tau = 4e-8
        variable_a = 0.22

        host_star_right_ascension = self.host_star_right_ascension if self.host_star_right_ascension is not None else self._phringe._scene.star.right_ascension
        host_star_declination = self.host_star_declination if self.host_star_declination is not None else self._phringe._scene.star.declination
        solar_ecliptic_latitude = self._solar_ecliptic_latitude if self._solar_ecliptic_latitude is not None else self._phringe._observation.solar_ecliptic_latitude

        ecliptic_latitude, relative_ecliptic_longitude = self._get_ecliptic_coordinates(
            host_star_right_ascension,
            host_star_declination,
            solar_ecliptic_latitude
        )
        spectral_flux_density = (
                variable_tau *
                (
                        get_blackbody_spectrum_standard_units(265,
                                                              self._phringe._instrument.wavelength_bin_centers) * self._solid_angle
                        + variable_a
                        * get_blackbody_spectrum_standard_units(5778,
                                                                self._phringe._instrument.wavelength_bin_centers) * self._solid_angle
                        * ((1 * u.Rsun).to(u.au) / (1.5 * u.au)).value ** 2
                ) *
                ((torch.pi / torch.arccos(torch.cos(torch.tensor(relative_ecliptic_longitude)) * torch.cos(
                    torch.tensor(ecliptic_latitude))))
                 / (torch.sin(torch.tensor(ecliptic_latitude)) ** 2 + 0.6 * (
                                self._phringe._instrument.wavelength_bin_centers / (11e-6)) ** (
                        -0.4) * torch.cos(
                            torch.tensor(ecliptic_latitude)) ** 2)) ** 0.5)
        return spectral_flux_density

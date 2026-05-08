from typing import Tuple, Union, Any

import torch
from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_si_units


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

        Parameters
        ----------
        value : Any
            The value given as input.
        info : ValidationInfo
            The validation information object.

        Returns
        -------
        float
            The right ascension in units of degrees.
        """
        return validate_quantity_units(
            value=value,
            field_name=info.field_name,
            unit_equivalency=(u.deg,)
        )

    @field_validator('host_star_declination')
    def _validate_declination(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the declination input.

        Parameters
        ----------
        value : Any
            The value given as input.
        info : ValidationInfo
            The validation information object.

        Returns
        -------
        float
            The declination in units of degrees.
        """
        return validate_quantity_units(
            value=value,
            field_name=info.field_name,
            unit_equivalency=(u.deg,)
        )

    def _get_ecliptic_coordinates(
            self,
            star_right_ascension,
            star_declination,
            solar_ecliptic_latitude
    ) -> Tuple:
        """Return the ecliptic coordinates corresponding to the star position in the sky.

        Parameters
        ----------
        star_right_ascension : float
            The right ascension of the star in units of radians.
        star_declination : float
            The declination of the star in units of radians.
        solar_ecliptic_latitude : float
            The solar ecliptic latitude in units of radians.

        Returns
        -------
        tuple
            Tuple containing the ecliptic latitude and relative ecliptic longitude in units of radians.
        """
        coordinates = SkyCoord(ra=star_right_ascension * u.rad, dec=star_declination * u.rad, frame='icrs')
        coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
        ecliptic_latitude = coordinates_ecliptic.lat.to(u.rad).value
        ecliptic_longitude = coordinates_ecliptic.lon.to(u.rad).value
        relative_ecliptic_longitude = ecliptic_longitude - solar_ecliptic_latitude

        return ecliptic_latitude, relative_ecliptic_longitude

    @property
    def sky_brightness_distribution(self) -> Tensor:
        sky_brightness_distribution = self.spectral_energy_distribution

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
        tau = 4e-8
        a = 0.22
        wavelengths = self._phringe._instrument.wavelength_bin_centers

        host_star_right_ascension = (
            self.host_star_right_ascension
            if self.host_star_right_ascension is not None
            else self._phringe._scene.star.right_ascension
        )
        host_star_declination = (
            self.host_star_declination
            if self.host_star_declination is not None
            else self._phringe._scene.star.declination
        )
        solar_ecliptic_latitude = (
            self._solar_ecliptic_latitude
            if self._solar_ecliptic_latitude is not None
            else self._phringe._observation.solar_ecliptic_latitude
        )

        ecliptic_latitude, relative_ecliptic_longitude = self._get_ecliptic_coordinates(
            host_star_right_ascension,
            host_star_declination,
            solar_ecliptic_latitude
        )

        spectral_energy_distribution = (
                tau * get_blackbody_spectrum_si_units(265, wavelengths) * self.solid_angle
                + a * get_blackbody_spectrum_si_units(5778, wavelengths) * self.solid_angle
                * ((1 * u.Rsun).to(u.au) / (1.5 * u.au)).value ** 2
                * ((torch.pi / torch.arccos(torch.cos(torch.tensor(relative_ecliptic_longitude))
                                            * torch.cos(torch.tensor(ecliptic_latitude))))
                   / (torch.sin(torch.tensor(ecliptic_latitude)) ** 2
                      + 0.6 * (wavelengths / (11e-6)) ** (-0.4)
                      * torch.cos(torch.tensor(ecliptic_latitude)) ** 2
                      )
                   ) ** 0.5)

        # Broadcast to grid dimension
        return spectral_energy_distribution[:, None, None]

from typing import Any, Union

import numpy as np
import torch
from astropy import units as u
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy.constants import sigma
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_standard_units


class Star(BaseSource):
    """Class representation of a star.

    Parameters
    ----------
    distance : str, float, or Quantity
        Distance between the host star and the instrument in units of length.
    mass : str, float, or Quantity
        Mass of the host star in units of weight.
    radius : str, float, or Quantity
        Radius of the host star in units of length.
    temperature : str, float, or Quantity
        Temperature of the host star in units of temperature.
    right_ascension : str, float, or Quantity
        Right ascension of the host star in units of degrees.
    declination : str, float, or Quantity
        Declination of the host star in units of degrees.
    """
    name: str
    distance: Union[str, float, Quantity]
    mass: Union[str, float, Quantity]
    radius: Union[str, float, Quantity]
    temperature: Union[str, float, Quantity]
    right_ascension: Union[str, float, Quantity]
    declination: Union[str, float, Quantity]

    @field_validator('distance')
    def _validate_distance(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the distance input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The distance in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,))

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('temperature')
    def _validate_temperature(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the temperature input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The temperature in units of temperature
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.K,))

    @field_validator('right_ascension')
    def _validate_right_ascension(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the right ascension input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The right ascension in units of seconds
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('declination')
    def _validate_declination(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the declination input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The declination in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @property
    def _angular_radius(self) -> float:
        return self.radius / self.distance

    @property
    def _habitable_zone_central_angular_radius(self) -> float:
        """Return the central habitable zone radius in angular units.

        :return: The central habitable zone radius in angular units
        """
        return self._habitable_zone_central_radius / self.distance

    @property
    def _habitable_zone_central_radius(self) -> float:
        """Return the central habitable zone radius of the star. Calculated as defined in Kopparapu et al. 2013.

        :return: The central habitable zone radius
        """
        incident_solar_flux_inner, incident_solar_flux_outer = 1.7665, 0.3240
        parameter_a_inner, parameter_a_outer = 1.3351E-4, 5.3221E-5
        parameter_b_inner, parameter_b_outer = 3.1515E-9, 1.4288E-9
        parameter_c_inner, parameter_c_outer = -3.3488E-12, -1.1049E-12
        temperature_difference = self.temperature - 5780

        incident_stellar_flux_inner = (incident_solar_flux_inner + parameter_a_inner * temperature_difference
                                       + parameter_b_inner * temperature_difference ** 2 + parameter_c_inner
                                       * temperature_difference ** 3)
        incident_stellar_flux_outer = (incident_solar_flux_outer + parameter_a_outer * temperature_difference
                                       + parameter_b_outer * temperature_difference ** 2 + parameter_c_outer
                                       * temperature_difference ** 3)

        radius_inner = np.sqrt(self.luminosity / 3.86e26 / incident_stellar_flux_inner)
        radius_outer = np.sqrt(self.luminosity / 3.86e26 / incident_stellar_flux_outer)
        return ((radius_outer + radius_inner) / 2 * u.au).si.value

    @property
    def luminosity(self):
        return 4 * np.pi * self.radius ** 2 * sigma * self.temperature ** 4

    @property
    def _sky_brightness_distribution(self) -> np.ndarray:
        number_of_wavelength_steps = len(self._phringe._instrument.wavelength_bin_centers)
        sky_brightness_distribution = torch.zeros(
            (number_of_wavelength_steps, self._phringe._grid_size, self._phringe._grid_size),
            device=self._phringe._device)
        radius_map = (torch.sqrt(self._sky_coordinates[0] ** 2 + self._sky_coordinates[1] ** 2) <= self._angular_radius)

        for index_wavelength in range(len(self._spectral_energy_distribution)):
            sky_brightness_distribution[index_wavelength] = radius_map * self._spectral_energy_distribution[
                index_wavelength]

        return sky_brightness_distribution

    @property
    def _sky_coordinates(self) -> Tensor:
        sky_coordinates = get_meshgrid(2 * (1.05 * self._angular_radius), self._phringe._grid_size,
                                       device=self._phringe._device)
        return torch.stack((sky_coordinates[0], sky_coordinates[1]))

    @property
    def _solid_angle(self):
        return np.pi * (self.radius / self.distance) ** 2

    @property
    def _spectral_energy_distribution(self) -> Tensor:
        return get_blackbody_spectrum_standard_units(
            self.temperature,
            self._phringe._instrument.wavelength_bin_centers
        ) * self._solid_angle

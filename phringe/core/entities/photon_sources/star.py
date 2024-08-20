from functools import cached_property
from typing import Any

import numpy as np
import torch
from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.io.validators import validate_quantity_units
from phringe.util.grid import get_meshgrid
from phringe.util.helpers import Coordinates
from phringe.util.spectrum import create_blackbody_spectrum


class Star(BasePhotonSource, BaseModel):
    """Class representation of a star.

    :param name: The name of the star
    :param distance: The distance to the star
    :param mass: The mass of the star
    :param radius: The radius of the star
    :param temperature: The temperature of the star
    :param luminosity: The luminosity of the star
    :param right_ascension: The right ascension of the star
    :param declination: The declination of the star
    """
    name: str
    distance: str
    mass: str
    radius: str
    temperature: str
    luminosity: str
    right_ascension: str
    declination: str

    @field_validator('distance')
    def _validate_distance(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the distance input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The distance in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('luminosity')
    def _validate_luminosity(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the luminosity input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The luminosity in units of luminosity
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.Lsun,)).to(
            u.Lsun).value

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,)).si.value

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('temperature')
    def _validate_temperature(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the temperature input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The temperature in units of temperature
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.K,)).si.value

    @cached_property
    def angular_radius(self) -> float:
        """Return the solid angle covered by the star on the sky.

        :return: The solid angle
        """
        return self.radius / self.distance

    @cached_property
    def habitable_zone_central_angular_radius(self) -> float:
        """Return the central habitable zone radius in angular units.

        :return: The central habitable zone radius in angular units
        """
        return self.habitable_zone_central_radius / self.distance

    @cached_property
    def habitable_zone_central_radius(self) -> float:
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

        radius_inner = np.sqrt(self.luminosity / incident_stellar_flux_inner)
        radius_outer = np.sqrt(self.luminosity / incident_stellar_flux_outer)
        return ((radius_outer + radius_inner) / 2 * u.au).si.value

    def _get_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        number_of_wavelength_steps = kwargs['number_of_wavelength_steps']
        sky_brightness_distribution = torch.zeros((number_of_wavelength_steps, grid_size, grid_size))
        radius_map = (torch.sqrt(self.sky_coordinates[0] ** 2 + self.sky_coordinates[1] ** 2) <= self.angular_radius)

        for index_wavelength in range(len(self.spectral_flux_density)):
            sky_brightness_distribution[index_wavelength] = radius_map * self.spectral_flux_density[
                index_wavelength]

        return sky_brightness_distribution

    def _get_sky_coordinates(self, grid_size, **kwargs) -> Coordinates:
        """Return the sky coordinate maps of the source1. The intensity responses are calculated in a resolution that
        allows the source1 to fill the grid, thus, each source1 needs to define its own sky coordinate map. Add 10% to the
        angular radius to account for rounding issues and make sure the source1 is fully covered within the map.

        :param grid_size: The grid size
        :return: A coordinates object containing the x- and y-sky coordinate maps
        """
        sky_coordinates = get_meshgrid(2 * (1.05 * self.angular_radius), grid_size)
        return torch.stack((sky_coordinates[0], sky_coordinates[1]))

    def _get_solid_angle(self, **kwargs) -> float:
        """Return the solid angle of the source1 object.

        :return: The solid angle
        """
        return np.pi * (self.radius / self.distance) ** 2

    def _get_spectral_flux_density(
            self,
            wavelength_steps: Tensor,
            grid_size: int,
            **kwargs
    ) -> Tensor:
        return create_blackbody_spectrum(self.temperature, wavelength_steps) * self.solid_angle

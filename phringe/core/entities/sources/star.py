import warnings
from functools import cached_property
from typing import Any

import numpy as np
import torch
from astropy import units as u
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.sources.base_source import CachedAttributesSource
from phringe.io.validators import validate_quantity_units
from phringe.util.grid import get_meshgrid
from phringe.util.helpers import Coordinates
from phringe.util.spectrum import create_blackbody_spectrum
from phringe.util.warning import MissingRequirementWarning


class Star(CachedAttributesSource):
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
    luminosity: str  # TODO: remove luminosity and calculate from radius and temperature
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

    @property
    def angular_radius(self) -> float:
        """Return the solid angle covered by the star on the sky.

        :return: The solid angle
        """
        return self._get_cached_value(
            attribute_name='angular_radius',
            compute_func=self._get_angular_radius,
            required_attributes=(
                self.radius,
                self.distance
            )
        )

    def _get_angular_radius(self):
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

    @property
    def _sky_brightness_distribution(self) -> np.ndarray:
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
                self._grid_size
            )
        )

    def _get_sky_brightness_distribution(self):
        number_of_wavelength_steps = len(self._instrument.wavelength_bin_centers)
        sky_brightness_distribution = torch.zeros((number_of_wavelength_steps, self._grid_size, self._grid_size),
                                                  device=self._device)
        radius_map = (torch.sqrt(self._sky_coordinates[0] ** 2 + self._sky_coordinates[1] ** 2) <= self.angular_radius)

        for index_wavelength in range(len(self._spectral_energy_distribution)):
            sky_brightness_distribution[index_wavelength] = radius_map * self._spectral_energy_distribution[
                index_wavelength]

        return sky_brightness_distribution

    @property
    def _sky_coordinates(self) -> Coordinates:
        """Return the sky coordinate maps of the source1. The intensity responses are calculated in a resolution that
        allows the source1 to fill the grid, thus, each source1 needs to define its own sky coordinate map. Add 10% to the
        angular radius to account for rounding issues and make sure the source1 is fully covered within the map.

        :param grid_size: The grid size
        :return: A coordinates object containing the x- and y-sky coordinate maps
        """
        if self._grid_size is None:
            warnings.warn(MissingRequirementWarning('Grid size', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='sky_coordinates',
            compute_func=self._get_sky_coordinates,
            required_attributes=(
                self._grid_size,
                self.angular_radius
            )
        )

    def _get_sky_coordinates(self):
        sky_coordinates = get_meshgrid(2 * (1.05 * self.angular_radius), self._grid_size)
        return torch.stack((sky_coordinates[0], sky_coordinates[1]))

    @property
    def _solid_angle(self):
        return self._get_cached_value(
            attribute_name='solid_angle',
            compute_func=self._get_solid_angle,
            required_attributes=(
                self.radius,
                self.distance
            )
        )

    def _get_solid_angle(self):
        return np.pi * (self.radius / self.distance) ** 2

    @property
    def _spectral_energy_distribution(self) -> Tensor:
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='spectral_energy_distribution',
            compute_func=self._get_spectral_energy_distribution,
            required_attributes=(
                self._instrument,
                self.radius,
                self.temperature,
                self._solid_angle
            )
        )

    def _get_spectral_energy_distribution(self) -> Tensor:
        return create_blackbody_spectrum(self.temperature,
                                         self._instrument.wavelength_bin_centers) * self._solid_angle

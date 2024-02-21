from typing import Any, Tuple

import astropy
import numpy as np
from astropy import units as u
from astropy.constants.codata2018 import G
from astropy.units import Quantity
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.io.validators import validate_quantity_units
from phringe.util.blackbody import create_blackbody_spectrum
from phringe.util.grid import get_index_of_closest_value, get_meshgrid
from phringe.util.helpers import Coordinates


class Planet(BasePhotonSource, BaseModel):
    """Class representation of a planet.

    :param name: The name of the planet
    :param mass: The mass of the planet
    :param radius: The radius of the planet
    :param temperature: The temperature of the planet
    :param semi_major_axis: The semi-major axis of the planet
    :param eccentricity: The eccentricity of the planet
    :param inclination: The inclination of the planet
    :param raan: The right ascension of the ascending node of the planet
    :param argument_of_periapsis: The argument of periapsis of the planet
    :param true_anomaly: The true anomaly of the planet
    :param angular_separation_from_star_x: The angular separation of the planet from the star in x-direction
    :param angular_separation_from_star_y: The angular separation of the planet from the star in y-direction
    """
    name: str
    mass: str
    radius: str
    temperature: str
    semi_major_axis: str
    eccentricity: float
    inclination: str
    raan: str
    argument_of_periapsis: str
    true_anomaly: str
    angular_separation_from_star_x: Any = None
    angular_separation_from_star_y: Any = None

    @field_validator('argument_of_periapsis')
    def _validate_argument_of_periapsis(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the argument of periapsis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The argument of periapsis in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('inclination')
    def _validate_inclination(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the inclination input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The inclination in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,))

    @field_validator('raan')
    def _validate_raan(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the raan input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The raan in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('semi_major_axis')
    def _validate_semi_major_axis(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the semi-major axis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The semi-major axis in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('temperature')
    def _validate_temperature(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the temperature input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The temperature in units of temperature
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.K,))

    @field_validator('true_anomaly')
    def _validate_true_anomaly(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the true anomaly input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The true anomaly in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    def _calculate_mean_spectral_flux_density(
            self,
            wavelength_steps: np.ndarray,
            grid_size: int,
            **kwargs
    ) -> np.ndarray:
        star_distance = kwargs.get('star_distance')
        solid_angle = np.pi * (self.radius.to(u.m) / (star_distance.to(u.m)) * u.rad) ** 2

        return create_blackbody_spectrum(self.temperature, wavelength_steps, solid_angle)

    def _calculate_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        """Calculate and return the sky brightness distribution.

        :param context: The context
        :return: The sky brightness distribution
        """
        has_planet_orbital_motion = kwargs.get('has_planet_orbital_motion')
        number_of_wavelength_steps = kwargs.get('number_of_wavelength_steps')

        if has_planet_orbital_motion:
            sky_brightness_distribution = np.zeros((len(self.sky_coordinates), number_of_wavelength_steps, grid_size,
                                                    grid_size)) * self.mean_spectral_flux_density[0].unit
            for index_sc, sky_coordinates in enumerate(self.sky_coordinates):
                index_x = get_index_of_closest_value(
                    sky_coordinates.x[0, :],
                    self.angular_separation_from_star_x[index_sc]
                )
                index_y = get_index_of_closest_value(
                    sky_coordinates.y[:, 0],
                    self.angular_separation_from_star_y[index_sc]
                )
                sky_brightness_distribution[index_sc, :, index_y, index_x] = self.mean_spectral_flux_density
        else:
            sky_brightness_distribution = np.zeros(
                (number_of_wavelength_steps, grid_size, grid_size)) * self.mean_spectral_flux_density.unit
            index_x = get_index_of_closest_value(self.sky_coordinates.x[0, :], self.angular_separation_from_star_x[0])
            index_y = get_index_of_closest_value(self.sky_coordinates.y[:, 0], self.angular_separation_from_star_y[0])
            sky_brightness_distribution[:, index_y, index_x] = self.mean_spectral_flux_density
        return sky_brightness_distribution

    def _calculate_sky_coordinates(self, grid_size, **kwargs) -> Coordinates:
        """Calculate and return the sky coordinates of the planet. Choose the maximum extent of the sky coordinates such
        that a circle with the radius of the planet's separation lies well (i.e. + 2x 20%) within the map. The construction
        of such a circle will be important to estimate the noise during signal extraction.

        :param grid_size: The grid size
        :param kwargs: The keyword arguments
        :return: The sky coordinates
        """
        time_steps = kwargs.get('time_steps')
        has_planet_orbital_motion = kwargs.get('has_planet_orbital_motion')
        star_distance = kwargs.get('star_distance')
        star_mass = kwargs.get('star_mass')
        self.angular_separation_from_star_x = np.zeros(len(time_steps)) * u.arcsec
        self.angular_separation_from_star_y = np.zeros(len(time_steps)) * u.arcsec

        # If planet motion is being considered, then the sky coordinates may change with each time step and thus
        # coordinates are created for each time step, rather than just once
        if has_planet_orbital_motion:
            sky_coordinates = np.zeros((len(time_steps)), dtype=object)
            for index_time, time_step in enumerate(time_steps):
                sky_coordinates[index_time] = self._get_coordinates(
                    grid_size,
                    time_step,
                    index_time,
                    has_planet_orbital_motion,
                    star_distance,
                    star_mass
                )
            return sky_coordinates
        else:
            return self._get_coordinates(grid_size, time_steps[0], 0, has_planet_orbital_motion, star_distance,
                                         star_mass)

    def _get_coordinates(
            self,
            grid_size: int,
            time_step: Quantity,
            index_time: int,
            has_planet_orbital_motion: bool,
            star_distance: Quantity,
            star_mass: Quantity
    ) -> Coordinates:
        """Return the sky coordinates of the planet.

        :param grid_size: The grid size
        :param time_step: The time step
        :param index_time: The index of the time step
        :param has_planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_distance: The distance of the star
        :param star_mass: The mass of the star
        :return: The sky coordinates
        """
        self.angular_separation_from_star_x[index_time], self.angular_separation_from_star_y[index_time] = (
            self._get_x_y_angular_separation_from_star(time_step, has_planet_orbital_motion, star_distance, star_mass))

        angular_radius = np.sqrt(
            self.angular_separation_from_star_x[index_time] ** 2 + self.angular_separation_from_star_y[index_time] ** 2)

        sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * angular_radius), grid_size)

        return Coordinates(sky_coordinates_at_time_step[0], sky_coordinates_at_time_step[1])

    def _get_x_y_separation_from_star(
            self,
            time_step: Quantity,
            has_planet_orbital_motion: bool,
            star_mass: Quantity
    ) -> Tuple:
        """Return the separation of the planet from the star in x- and y-direction. If the planet orbital motion is
        considered, calculate the new position for each time step.

        :param time_step: The time step
        :param has_planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_mass: The mass of the star
        :return: A tuple containing the x- and y- coordinates
        """
        star = Body(parent=None, k=G * (star_mass + self.mass), name='Star')
        orbit = Orbit.from_classical(star, a=self.semi_major_axis, ecc=u.Quantity(self.eccentricity),
                                     inc=self.inclination,
                                     raan=self.raan,
                                     argp=self.argument_of_periapsis, nu=self.true_anomaly)
        if has_planet_orbital_motion:
            orbit_propagated = orbit.propagate(time_step)
        else:
            orbit_propagated = orbit
        return (orbit_propagated.r[0], orbit_propagated.r[1])

    def _get_x_y_angular_separation_from_star(
            self,
            time_step: Quantity,
            planet_orbital_motion: bool,
            star_distance: Quantity,
            star_mass: Quantity
    ) -> Tuple:
        """Return the angular separation of the planet from the star in x- and y-direction.

        :param time_step: The time step
        :param planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_distance: The distance of the star
        :param star_mass: The mass of the star
        :return: A tuple containing the x- and y- coordinates
        """
        separation_from_star_x, separation_from_star_y = self._get_x_y_separation_from_star(time_step,
                                                                                            planet_orbital_motion,
                                                                                            star_mass)
        angular_separation_from_star_x = ((separation_from_star_x.to(u.m) / star_distance.to(u.m)) * u.rad).to(
            u.arcsec)
        angular_separation_from_star_y = ((separation_from_star_y.to(u.m) / star_distance.to(u.m)) * u.rad).to(
            u.arcsec)
        return (angular_separation_from_star_x, angular_separation_from_star_y)

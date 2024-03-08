import time
from typing import Any, Tuple

import astropy
import numpy as np
import spectres
import torch
from astropy import units as u
from astropy.constants.codata2018 import G
from astropy.units import Quantity
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.io.validators import validate_quantity_units
from phringe.util.grid import get_index_of_closest_value, get_meshgrid, get_index_of_closest_value_numpy
from phringe.util.spectrum import create_blackbody_spectrum


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

    def _calculate_mean_spectral_flux_density(self, wavelength_steps: np.ndarray, grid_size: int,
                                              **kwargs) -> np.ndarray:
        """Bin the already generated blackbody spectra / loaded input spectra of the planet to the wavelength steps of
        the simulation.

        :param wavelength_steps: The wavelength steps
        :return: The binned mean spectral flux density
        """
        maximum_wavelength_steps = kwargs.get('maximum_wavelength_steps')

        t0 = time.time_ns()
        binned_mean_spectral_flux_density = spectres.spectres(
            wavelength_steps.to(u.um).value,
            maximum_wavelength_steps,
            self.mean_spectral_flux_density.value,
            fill=0
        ) * self.mean_spectral_flux_density.unit

        t1 = time.time_ns()
        print(f"Time to bin mean spectral flux density: {(t1 - t0) / 1e9} s")
        return binned_mean_spectral_flux_density

    def _calculate_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        """Calculate and return the sky brightness distribution.

        :param context: The context
        :return: The sky brightness distribution
        """
        has_planet_orbital_motion = kwargs.get('has_planet_orbital_motion')
        number_of_wavelength_steps = kwargs.get('number_of_wavelength_steps')

        if has_planet_orbital_motion:
            sky_brightness_distribution = np.zeros((len(self.sky_coordinates[1]), number_of_wavelength_steps, grid_size,
                                                    grid_size)) * self.mean_spectral_flux_density[0].unit
            for index_time in range(len(self.sky_coordinates[1])):
                sky_coordinates = self.sky_coordinates[:, index_time]
                index_x = get_index_of_closest_value_numpy(
                    sky_coordinates[0, 0, :].value,
                    self.angular_separation_from_star_x[index_time].to(u.rad).value
                )
                index_y = get_index_of_closest_value_numpy(
                    sky_coordinates[1, :, 0].value,
                    self.angular_separation_from_star_y[index_time].to(u.rad).value
                )
                sky_brightness_distribution[index_time, :, index_y, index_x] = self.mean_spectral_flux_density
        else:
            sky_brightness_distribution = np.zeros(
                (number_of_wavelength_steps, grid_size, grid_size)) * self.mean_spectral_flux_density.unit
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            index_x = get_index_of_closest_value(torch.asarray(self.sky_coordinates[0, 0, :].value),
                                                 self.angular_separation_from_star_x[0].value)
            index_y = get_index_of_closest_value(torch.asarray(self.sky_coordinates[1, :, 0].value),
                                                 self.angular_separation_from_star_y[0].value)
            sky_brightness_distribution[:, index_y, index_x] = self.mean_spectral_flux_density
        return sky_brightness_distribution

    def _calculate_sky_coordinates(self, grid_size, **kwargs) -> np.ndarray:
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
            sky_coordinates = np.zeros((2, len(time_steps), grid_size, grid_size))
            for index_time, time_step in enumerate(time_steps):
                sky_coordinates[:, index_time] = self._get_coordinates(
                    grid_size,
                    time_step,
                    index_time,
                    has_planet_orbital_motion,
                    star_distance,
                    star_mass
                )
            return sky_coordinates * u.rad
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
    ) -> np.ndarray:
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
            self._get_x_y_angular_separation_from_star(time_step, has_planet_orbital_motion, star_distance,
                                                       star_mass))

        angular_radius = np.sqrt(
            self.angular_separation_from_star_x[index_time] ** 2 + self.angular_separation_from_star_y[
                index_time] ** 2).to(u.rad)

        sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * angular_radius), grid_size)

        return np.stack((sky_coordinates_at_time_step[0], sky_coordinates_at_time_step[1]))

    def orbital_elements_to_sky_position(self, a, e, i, Omega, omega, nu):
        # Convert angles from degrees to radians
        # i = np.radians(i)
        # Omega = np.radians(Omega)
        # omega = np.radians(omega)
        # nu = np.radians(nu)
        # https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf

        M = np.arctan2(-np.sqrt(1 - e ** 2) * np.sin(nu), -e - np.cos(nu)) + np.pi - e * (
                np.sqrt(1 - e ** 2) * np.sin(nu)) / (1 + e * np.cos(nu))

        E = M
        for _ in range(10):  # Newton's method iteration
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

        # nu2 = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

        r = a * (1 - e * np.cos(E))

        # Position in the orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)

        x = x_orb * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i)) - y_orb * (
                np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.sin(Omega) * np.cos(i))
        y = x_orb * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i)) + y_orb * (
                np.cos(omega) * np.cos(Omega) * np.cos(i) - np.sin(omega) * np.sin(Omega))

        # x_temp = x_orb * np.cos(omega) - y_orb * np.sin(omega)
        # y_temp = x_orb * np.sin(omega) + y_orb * np.cos(omega)
        # z_temp = 0  # Initial z-position is 0 since in orbital plane
        #
        # # Rotate by inclination (i)
        # x_inclined = x_temp
        # y_inclined = y_temp * np.cos(i)
        # z_inclined = y_temp * np.sin(i)
        #
        # # Rotate by longitude of ascending node (Omega)
        # x_final = x_inclined * np.cos(Omega) - z_inclined * np.sin(Omega)
        # y_final = y_inclined
        # z_final = x_inclined * np.sin(Omega) + z_inclined * np.cos(Omega)

        # For sky projection, we are generally interested in the x and y components
        return x, y

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
        # t0 = time.time_ns()
        star = Body(parent=None, k=G * (star_mass + self.mass), name='Star')
        orbit = Orbit.from_classical(star, a=self.semi_major_axis, ecc=u.Quantity(self.eccentricity),
                                     inc=self.inclination,
                                     raan=self.raan,
                                     argp=self.argument_of_periapsis, nu=self.true_anomaly)
        # t1 = time.time_ns()
        # print(f"Time to set up orbit: {(t1 - t0) / 1e9} s")
        if has_planet_orbital_motion:
            # orbit_propagated = orbit.propagate(time_step)
            pass
        else:
            # orbit_propagated = orbit
            # t0 = time.time_ns()
            # a = (
            #     (self.semi_major_axis * (1 - self.eccentricity ** 2)).to(u.km),
            #     Quantity(self.eccentricity),
            #     self.inclination.to(u.rad),
            #     self.raan.to(u.rad),
            #     self.argument_of_periapsis.to(u.rad),
            #     self.true_anomaly.to(u.rad),
            # )
            # r, v = coe2rv(
            #     (G * (star_mass + self.mass)).to(u.km ** 3 / u.s ** 2), *a
            # )
            # t1 = time.time_ns()
            # print(f"Time to create orbit: {(t1 - t0) / 1e9} s")

            # Example usage
            a = self.semi_major_axis.to(u.m).value  # Semi-major axis
            e = self.eccentricity  # Eccentricity
            i = self.inclination.to(u.rad).value  # Inclination in degrees
            Omega = self.raan.to(u.rad).value  # Longitude of the ascending node in degrees
            omega = self.argument_of_periapsis.to(u.rad).value  # Argument of periapsis in degrees
            M = self.true_anomaly.to(u.rad).value  # Mean anomaly in degrees
            # t0 = time.time_ns()

            x1, y1 = self.orbital_elements_to_sky_position(a, e, i, Omega, omega, M)
            # t1 = time.time_ns()
            # print(f"Time to create orbit 2: {(t1 - t0) / 1e9} s")
            #
            # x = (orbit_propagated.r[0], orbit_propagated.r[1], orbit_propagated.r[2])
            # print(x)
            # print(x1, y1)
        return x1 * u.m, y1 * u.m
        # print(x)
        # return x

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

    def calculate_blackbody_spectrum(
            self,
            wavelength_steps: np.ndarray,
            **kwargs
    ) -> np.ndarray:
        star_distance = kwargs.get('star_distance')
        solid_angle = np.pi * (self.radius.to(u.m) / (star_distance.to(u.m)) * u.rad) ** 2

        a = create_blackbody_spectrum(self.temperature, wavelength_steps, solid_angle)

        return a

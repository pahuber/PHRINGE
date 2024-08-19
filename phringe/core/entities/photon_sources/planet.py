from typing import Any, Tuple

import numpy as np
import spectres
import torch
from astropy import units as u
from astropy.constants.codata2018 import G
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.entities.photon_sources.base_photon_source import BasePhotonSource
from phringe.io.txt_reader import TXTReader
from phringe.io.validators import validate_quantity_units
from phringe.util.grid import get_index_of_closest_value, get_meshgrid
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
    :param grid_position: The grid position of the planet
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
    path_to_spectrum: Any
    angular_separation_from_star_x: Any = None
    angular_separation_from_star_y: Any = None
    grid_position: Tuple = None

    @field_validator('argument_of_periapsis')
    def _validate_argument_of_periapsis(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the argument of periapsis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The argument of periapsis in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,)).si.value

    @field_validator('inclination')
    def _validate_inclination(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the inclination input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The inclination in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,)).si.value

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,)).si.value

    @field_validator('raan')
    def _validate_raan(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the raan input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The raan in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,)).si.value

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('semi_major_axis')
    def _validate_semi_major_axis(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the semi-major axis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The semi-major axis in units of length
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

    @field_validator('true_anomaly')
    def _validate_true_anomaly(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the true anomaly input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The true anomaly in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,)).si.value

    def _get_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        """Calculate and return the sky brightness distribution.

        :param context: The context
        :return: The sky brightness distribution
        """
        has_planet_orbital_motion = kwargs.get('has_planet_orbital_motion')
        number_of_wavelength_steps = kwargs.get('number_of_wavelength_steps')

        if has_planet_orbital_motion:
            sky_brightness_distribution = torch.zeros(
                (len(self.sky_coordinates[1]), number_of_wavelength_steps, grid_size,
                 grid_size))
            for index_time in range(len(self.sky_coordinates[1])):
                sky_coordinates = self.sky_coordinates[:, index_time]
                index_x = get_index_of_closest_value_numpy(
                    sky_coordinates[0, :, 0],
                    self.angular_separation_from_star_x[index_time]
                )
                index_y = get_index_of_closest_value_numpy(
                    sky_coordinates[1, 0, :],
                    self.angular_separation_from_star_y[index_time]
                )
                sky_brightness_distribution[index_time, :, index_x, index_y] = self.spectral_flux_density
        elif self.grid_position:
            sky_brightness_distribution = torch.zeros(
                (number_of_wavelength_steps, grid_size, grid_size))
            sky_brightness_distribution[:, self.grid_position[1], self.grid_position[0]] = self.spectral_flux_density
        else:
            sky_brightness_distribution = torch.zeros(
                (number_of_wavelength_steps, grid_size, grid_size))
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            index_x = get_index_of_closest_value(torch.asarray(self.sky_coordinates[0, :, 0]),
                                                 self.angular_separation_from_star_x[0])
            index_y = get_index_of_closest_value(torch.asarray(self.sky_coordinates[1, 0, :]),
                                                 self.angular_separation_from_star_y[0])
            sky_brightness_distribution[:, index_x, index_y] = self.spectral_flux_density
        return sky_brightness_distribution

    def _get_sky_coordinates(self, grid_size, **kwargs) -> np.ndarray:
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
        self.angular_separation_from_star_x = torch.zeros(len(time_steps))
        self.angular_separation_from_star_y = torch.zeros(len(time_steps))

        # If planet motion is being considered, then the sky coordinates may change with each time step and thus
        # coordinates are created for each time step, rather than just once
        if has_planet_orbital_motion:
            sky_coordinates = torch.zeros((2, len(time_steps), grid_size, grid_size))
            for index_time, time_step in enumerate(time_steps):
                sky_coordinates[:, index_time] = self._get_coordinates(
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

    def _get_solid_angle(self, **kwargs) -> float:
        """Calculate and return the solid angle of the planet.

        :param kwargs: The keyword arguments
        :return: The solid angle
        """
        star_distance = kwargs.get('star_distance')
        return torch.pi * (self.radius / star_distance) ** 2

    def _get_spectral_flux_density(self, wavelength_bin_centers: Tensor, grid_size: int, **kwargs) -> Tensor:
        """Calculate the spectral flux density of the planet in units of ph s-1 m-3. Use the previously generated
        reference spectrum in units of ph s-1 m-3 sr-1 and the solid angle to calculate it and bin it to the
        simulation wavelength bin centers.

        :param wavelength_bin_centers: The wavelength bin centers
        :param grid_size: The grid size
        :param kwargs: The keyword arguments
        :return: The binned mean spectral flux density in units of ph s-1 m-3
        """
        if self.path_to_spectrum:
            fluxes, wavelengths = TXTReader.read(self.path_to_spectrum)
            binned_spectral_flux_density = spectres.spectres(
                wavelength_bin_centers.numpy(),
                wavelengths.numpy(),
                fluxes.numpy(),
                fill=0,
                verbose=False
            ) * self.solid_angle
            return torch.asarray(binned_spectral_flux_density, dtype=torch.float32)
        else:
            binned_spectral_flux_density = torch.asarray(
                create_blackbody_spectrum(
                    self.temperature,
                    wavelength_bin_centers
                )
                , dtype=torch.float32) * self.solid_angle
            return binned_spectral_flux_density

    def _get_coordinates(
            self,
            grid_size: int,
            time_step: float,
            index_time: int,
            has_planet_orbital_motion: bool,
            star_distance: float,
            star_mass: float
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

        angular_radius = torch.sqrt(
            self.angular_separation_from_star_x[index_time] ** 2 + self.angular_separation_from_star_y[
                index_time] ** 2)

        sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * angular_radius), grid_size)

        return torch.stack((sky_coordinates_at_time_step[0], sky_coordinates_at_time_step[1]))

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
            time_step: float,
            has_planet_orbital_motion: bool,
            star_mass: float
    ) -> Tuple:
        """Return the separation of the planet from the star in x- and y-direction. If the planet orbital motion is
        considered, calculate the new position for each time step.

        :param time_step: The time step
        :param has_planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_mass: The mass of the star
        :return: A tuple containing the x- and y- coordinates
        """
        star = Body(parent=None, k=G * (star_mass + self.mass) * u.kg, name='Star')
        orbit = Orbit.from_classical(star, a=self.semi_major_axis * u.m, ecc=u.Quantity(self.eccentricity),
                                     inc=self.inclination * u.rad,
                                     raan=self.raan * u.rad,
                                     argp=self.argument_of_periapsis * u.rad, nu=self.true_anomaly * u.rad)
        if has_planet_orbital_motion:
            orbit_propagated = orbit.propagate(time_step * u.s)
            x, y = (orbit_propagated.r[0].to(u.m).value, orbit_propagated.r[1].to(u.m).value)
            pass
        else:
            a = self.semi_major_axis  # Semi-major axis
            e = self.eccentricity  # Eccentricity
            i = self.inclination  # Inclination in degrees
            Omega = self.raan  # Longitude of the ascending node in degrees
            omega = self.argument_of_periapsis  # Argument of periapsis in degrees
            M = self.true_anomaly  # Mean anomaly in degrees

            x, y = self.orbital_elements_to_sky_position(a, e, i, Omega, omega, M)
        return x, y

    def _get_x_y_angular_separation_from_star(
            self,
            time_step: float,
            planet_orbital_motion: bool,
            star_distance: float,
            star_mass: float
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
        angular_separation_from_star_x = separation_from_star_x / star_distance
        angular_separation_from_star_y = separation_from_star_y / star_distance
        return (angular_separation_from_star_x, angular_separation_from_star_y)

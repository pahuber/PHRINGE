import warnings
from typing import Any, Tuple, Union

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

from phringe.entities.sources.base_source import BaseSource
from phringe.io.txt_reader import TXTReader
from phringe.io.validators import validate_quantity_units
from phringe.util.grid import get_index_of_closest_value, get_meshgrid
from phringe.util.spectrum import create_blackbody_spectrum
from phringe.util.warning import MissingRequirementWarning


class Planet(BaseSource, BaseModel):
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
    :param _angular_separation_from_star_x: The angular separation of the planet from the star in x-direction
    :param _angular_separation_from_star_y: The angular separation of the planet from the star in y-direction
    :param grid_position: The grid position of the planet
    """
    name: str
    has_orbital_motion: bool
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
    grid_position: Tuple = None
    host_star_distance: Any = None
    host_star_mass: Any = None
    _angular_separation_from_star_x: Any = None
    _angular_separation_from_star_y: Any = None
    _simulation_time_steps: Any = None

    # TODO: add option to manually set host star mass and distance in case no star has been added to the scene

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

    @property
    def _sky_brightness_distribution(self) -> np.ndarray:
        """Calculate and return the sky brightness distribution.

        :param context: The context
        :return: The sky brightness distribution
        """
        has_planet_orbital_motion = False  # TODO: Implement this
        # number_of_wavelength_steps = kwargs.get('number_of_wavelength_steps')

        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_brightness_distribution'))
            return None

        if self._grid_size is None:
            warnings.warn(MissingRequirementWarning('grid size', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='sky_brightness_distribution',
            compute_func=self._get_sky_brightness_distribution,
            required_attributes=(
                self._instrument,
                self._grid_size
            )
        )

    @property
    def _sky_coordinates(self) -> Union[Tensor, None]:
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_coordinates'))
            return None

        if self._grid_size is None:
            warnings.warn(MissingRequirementWarning('grid size', 'sky_coordinates'))
            return None

        if self.host_star_mass is None:
            warnings.warn(MissingRequirementWarning('host star mass', 'sky_coordinates'))
            return None

        if self.host_star_distance is None:
            warnings.warn(MissingRequirementWarning('host star distance', 'sky_coordinates'))
            return None

        return self._get_cached_value(
            attribute_name='sky_coordinates',
            compute_func=self._get_sky_coordinates,
            required_attributes=(
                self._instrument,
                self._grid_size,
                self.has_orbital_motion,
                self._simulation_time_steps,
                self.host_star_distance,
                self.host_star_mass
            )
        )

    @property
    def _solid_angle(self):
        if self.host_star_distance is None:
            warnings.warn(MissingRequirementWarning('host star distance', 'solid_angle'))
            return None

        return self._get_cached_value(
            attribute_name='solid_angle',
            compute_func=self._get_solid_angle,
            required_attributes=(
                self.host_star_distance,
                self.radius
            )
        )

    @property
    def _spectral_energy_distribution(self) -> Union[Tensor, None]:
        if self._instrument is None:
            warnings.warn(MissingRequirementWarning('Instrument', 'sky_brightness_distribution'))
            return None

        if self._grid_size is None:
            warnings.warn(MissingRequirementWarning('grid size', 'sky_brightness_distribution'))
            return None

        return self._get_cached_value(
            attribute_name='spectral_energy_distribution',
            compute_func=self._get_spectral_energy_distribution,
            required_attributes=(
                self._instrument,
                self._grid_size,
                self.temperature,
                self.path_to_spectrum,
                self._solid_angle
            )
        )

    def _get_sky_brightness_distribution(self):

        # if has_planet_orbital_motion:
        #     sky_brightness_distribution = torch.zeros(
        #     (len(self.sky_coordinates[1]),
        #
        #     number_of_wavelength_steps, grid_size,
        #     grid_size))
        #     for index_time in range(len(self.sky_coordinates[1])):
        #         sky_coordinates = self.sky_coordinates[:, index_time]
        #     index_x = get_index_of_closest_value_numpy(
        #         sky_coordinates[0, :, 0],
        #         self.angular_separation_from_star_x[index_time]
        #     )
        #     index_y = get_index_of_closest_value_numpy(
        #         sky_coordinates[1, 0, :],
        #         self.angular_separation_from_star_y[index_time]
        #     )
        #     sky_brightness_distribution[index_time, :, index_x, index_y] = self.spectral_flux_density

        number_of_wavelength_steps = len(self._instrument.wavelength_bin_centers)
        if self.grid_position:
            sky_brightness_distribution = torch.zeros(
                (number_of_wavelength_steps, self._grid_size, self._grid_size), device=self._device)
            sky_brightness_distribution[:, self.grid_position[1],
            self.grid_position[0]] = self._spectral_energy_distribution
            self._angular_separation_from_star_x = self._sky_coordinates[
                0, self.grid_position[1], self.grid_position[0]]
            self._angular_separation_from_star_y = self._sky_coordinates[
                1, self.grid_position[1], self.grid_position[0]]
        else:
            sky_brightness_distribution = torch.zeros(
                (number_of_wavelength_steps, self._grid_size, self._grid_size), device=self._device)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            index_x = get_index_of_closest_value(torch.asarray(self._sky_coordinates[0, :, 0], device=self._device),
                                                 self._angular_separation_from_star_x[0])
            index_y = get_index_of_closest_value(torch.asarray(self._sky_coordinates[1, 0, :], device=self._device),
                                                 self._angular_separation_from_star_y[0])
            sky_brightness_distribution[:, index_x, index_y] = self._spectral_energy_distribution
        return sky_brightness_distribution

    def _get_sky_coordinates(self) -> np.ndarray:
        """Calculate and return the sky coordinates of the planet. Choose the maximum extent of the sky coordinates such
        that a circle with the radius of the planet's separation lies well (i.e. + 2x 20%) within the map. The construction
        of such a circle will be important to estimate the noise during signal extraction.

        :param grid_size: The grid size
        :param kwargs: The keyword arguments
        :return: The sky coordinates
        """
        self._angular_separation_from_star_x = torch.zeros(len(self._simulation_time_steps), device=self._device)
        self._angular_separation_from_star_y = torch.zeros(len(self._simulation_time_steps), device=self._device)

        # If planet motion is being considered, then the sky coordinates may change with each time step and thus
        # coordinates are created for each time step, rather than just once
        if self.has_orbital_motion:
            sky_coordinates = torch.zeros((2, len(self._simulation_time_steps), self._grid_size, self._grid_size),
                                          device=self._device)
            for index_time, time_step in enumerate(self._simulation_time_steps):
                sky_coordinates[:, index_time] = self._get_coordinates(
                    time_step,
                    index_time,
                )
            return sky_coordinates
        else:
            return self._get_coordinates(self._simulation_time_steps[0], 0)

    def _get_solid_angle(self) -> float:
        """Calculate and return the solid angle of the planet.

        :param kwargs: The keyword arguments
        :return: The solid angle
        """
        return torch.pi * (self.radius / self.host_star_distance) ** 2

    def _get_spectral_energy_distribution(self) -> Tensor:
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
                self._instrument.wavelength_bin_centers.numpy(),
                wavelengths.numpy(),
                fluxes.numpy(),
                fill=0,
                verbose=False
            ) * self._get_solid_angle
            return torch.asarray(binned_spectral_flux_density, dtype=torch.float32, device=self._device)
        else:
            binned_spectral_flux_density = torch.asarray(
                create_blackbody_spectrum(
                    self.temperature,
                    self._instrument.wavelength_bin_centers
                )
                , dtype=torch.float32, device=self._device) * self._solid_angle
            return binned_spectral_flux_density

    def _get_coordinates(
            self,
            time_step: float,
            index_time: int
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
        self._angular_separation_from_star_x[index_time], self._angular_separation_from_star_y[index_time] = (
            self._get_x_y_angular_separation_from_star(time_step)
        )

        angular_radius = torch.sqrt(
            self._angular_separation_from_star_x[index_time] ** 2
            + self._angular_separation_from_star_y[index_time] ** 2
        )

        sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * angular_radius), self._grid_size, device=self._device)

        return torch.stack((sky_coordinates_at_time_step[0], sky_coordinates_at_time_step[1]))

    def _convert_orbital_elements_to_sky_position(self, a, e, i, Omega, omega, nu):
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

        return x, y

    def _get_x_y_angular_separation_from_star(
            self,
            time_step: float
    ) -> Tuple:
        """Return the angular separation of the planet from the star in x- and y-direction.

        :param time_step: The time step
        :param planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_distance: The distance of the star
        :param star_mass: The mass of the star
        :return: A tuple containing the x- and y- coordinates
        """
        separation_from_star_x, separation_from_star_y = self._get_x_y_separation_from_star(time_step)
        angular_separation_from_star_x = separation_from_star_x / self.host_star_distance
        angular_separation_from_star_y = separation_from_star_y / self.host_star_distance
        return (angular_separation_from_star_x, angular_separation_from_star_y)

    def _get_x_y_separation_from_star(self, time_step: float, ) -> Tuple:
        """Return the separation of the planet from the star in x- and y-direction. If the planet orbital motion is
        considered, calculate the new position for each time step.

        :param time_step: The time step
        :param has_planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_mass: The mass of the star
        :return: A tuple containing the x- and y- coordinates
        """
        star = Body(parent=None, k=G * (self.host_star_mass + self.mass) * u.kg, name='Star')
        orbit = Orbit.from_classical(star, a=self.semi_major_axis * u.m, ecc=u.Quantity(self.eccentricity),
                                     inc=self.inclination * u.rad,
                                     raan=self.raan * u.rad,
                                     argp=self.argument_of_periapsis * u.rad, nu=self.true_anomaly * u.rad)
        if self.has_orbital_motion:
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

            x, y = self._convert_orbital_elements_to_sky_position(a, e, i, Omega, omega, M)
        return x, y

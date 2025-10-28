from typing import Any, Tuple, Union

import numpy as np
import torch
from astropy import units as u
from astropy.constants.codata2018 import G
from astropy.units import Quantity
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.io.input_spectrum import InputSpectrum
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_index_of_closest_value, get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_standard_units


def _convert_orbital_elements_to_sky_position(a, e, i, Omega, omega, nu):
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


class Planet(BaseSource):
    """Class representation of a planet.

    Parameters
    ----------
    has_orbital_motion: bool
        Whether the planet has orbital motion. If not, it is assumed to be static in its orbit throughout the simulation.
    mass: float or str or Quantity
        The mass of the planet in units of weight.
    radius: float or str or Quantity
        The radius of the planet in units of length.
    temperature: float or str or Quantity
        The effective temperature of the planet in units of temperature.
    semi_major_axis: float or str or Quantity
        The semi-major axis of the planet's orbit in units of length.
    eccentricity: float
        The eccentricity of the planet's orbit.
    inclination: float or str or Quantity
        The inclination of the planet's orbit in units of degrees.
    raan: float or str or Quantity
        The right ascension of the ascending node of the planet's orbit in units of degrees.
    argument_of_periapsis: float or str or Quantity
        The argument of periapsis of the planet's orbit in units of degrees.
    true_anomaly: float or str or Quantity
        The true anomaly of the planet's orbit in units of degrees.
    input_spectrum: InputSpectrum, optional
        The input spectrum of the planet. If None, a blackbody spectrum is generated.
    grid_position: Tuple[int, int] , optional
        The grid position of the planet in the sky. If None, the position is calculated from its orbital elements.
    host_star_distance: float or str or Quantity, optional
        The distance of the host star from the planet in units of length. Only required if no host star is specified in the scene.
    host_star_mass: float or str or Quantity, optional
        The mass of the host star in units of weight. Only required if no host star is specified in the scene.
    """
    name: str
    has_orbital_motion: bool
    mass: Union[str, float, Quantity]
    radius: Union[str, float, Quantity]
    temperature: Union[str, float, Quantity]
    semi_major_axis: Union[str, float, Quantity]
    eccentricity: float
    inclination: Union[str, float, Quantity]
    raan: Union[str, float, Quantity]
    argument_of_periapsis: Union[str, float, Quantity]
    true_anomaly: Union[str, float, Quantity]
    input_spectrum: Union[InputSpectrum, None]
    grid_position: Tuple = None
    host_star_distance: Union[str, float, Quantity] = None
    host_star_mass: Union[str, float, Quantity] = None
    _angular_separation_from_star_x: Any = None
    _angular_separation_from_star_y: Any = None
    _simulation_time_steps: Any = None

    @field_validator('argument_of_periapsis')
    def _validate_argument_of_periapsis(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the argument of periapsis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The argument of periapsis in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('inclination')
    def _validate_inclination(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the inclination input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The inclination in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,))

    @field_validator('raan')
    def _validate_raan(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the raan input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The raan in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the radius input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The radius in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('semi_major_axis')
    def _validate_semi_major_axis(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the semi-major axis input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The semi-major axis in units of length
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

    @field_validator('true_anomaly')
    def _validate_true_anomaly(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the true anomaly input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The true anomaly in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('host_star_distance')
    def _validate_host_star_distance(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the host star distance input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The host star distance in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('host_star_mass')
    def _validate_host_star_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the host star mass input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The host star mass in units of weight
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,))

    @property
    def _sky_brightness_distribution(self) -> np.ndarray:
        """Calculate and return the sky brightness distribution.

        :param context: The context
        :return: The sky brightness distribution
        """
        number_of_wavelength_steps = len(self._phringe._instrument.wavelength_bin_centers)

        if self.has_orbital_motion:
            sky_brightness_distribution = torch.zeros(
                (len(self._sky_coordinates[1]), number_of_wavelength_steps, self._phringe._grid_size,
                 self._phringe._grid_size),
                device=self._phringe._device)
            for index_time in range(len(self._sky_coordinates[1])):
                sky_coordinates = self._sky_coordinates[:, index_time]
                index_x = get_index_of_closest_value(
                    sky_coordinates[0, :, 0],
                    self._angular_separation_from_star_x[index_time]
                )
                index_y = get_index_of_closest_value(
                    sky_coordinates[1, 0, :],
                    self._angular_separation_from_star_y[index_time]
                )
                sky_brightness_distribution[index_time, :, index_x, index_y] = self._spectral_energy_distribution
        elif self.grid_position:
            sky_brightness_distribution = torch.zeros(
                (number_of_wavelength_steps, self._phringe._grid_size, self._phringe._grid_size),
                device=self._phringe._device)
            sky_brightness_distribution[:, self.grid_position[1],
            self.grid_position[0]] = self._spectral_energy_distribution
            self._angular_separation_from_star_x = self._sky_coordinates[
                0, self.grid_position[1], self.grid_position[0]]
            self._angular_separation_from_star_y = self._sky_coordinates[
                1, self.grid_position[1], self.grid_position[0]]
        else:
            sky_brightness_distribution = torch.zeros(
                (number_of_wavelength_steps, self._phringe._grid_size, self._phringe._grid_size),
                device=self._phringe._device)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            index_x = get_index_of_closest_value(
                torch.asarray(self._sky_coordinates[0, :, 0], device=self._phringe._device),
                self._angular_separation_from_star_x[0])
            index_y = get_index_of_closest_value(
                torch.asarray(self._sky_coordinates[1, 0, :], device=self._phringe._device),
                self._angular_separation_from_star_y[0])
            sky_brightness_distribution[:, index_x, index_y] = self._spectral_energy_distribution
        return sky_brightness_distribution

    @property
    def _sky_coordinates(self) -> Union[Tensor, None]:
        self._angular_separation_from_star_x = torch.zeros(len(self._phringe.simulation_time_steps),
                                                           device=self._phringe._device)
        self._angular_separation_from_star_y = torch.zeros(len(self._phringe.simulation_time_steps),
                                                           device=self._phringe._device)

        # If planet motion is being considered, then the sky coordinates may change with each time step and thus
        # coordinates are created for each time step, rather than just once
        if self.has_orbital_motion:
            sky_coordinates = torch.zeros(
                (2, len(self._phringe.simulation_time_steps), self._phringe._grid_size, self._phringe._grid_size),
                device=self._phringe._device
            )
            for index_time, time_step in enumerate(self._phringe.simulation_time_steps):
                sky_coordinates[:, index_time] = self._get_coordinates(
                    time_step.item(),
                    index_time,
                )
            return sky_coordinates
        else:
            return self._get_coordinates(self._phringe.simulation_time_steps[0], 0)

    @property
    def _solid_angle(self):
        host_star_distance = self.host_star_distance if self.host_star_distance is not None else self._phringe._scene.star.distance
        return torch.pi * (self.radius / host_star_distance) ** 2

    @property
    def _spectral_energy_distribution(self) -> Union[Tensor, None]:
        if self.input_spectrum is not None:
            return self.input_spectrum.get_spectral_energy_distribution(
                self._phringe._instrument.wavelength_bin_centers,
                self._solid_angle,
                self._phringe._device
            )

        # if self.path_to_spectrum:
        #     fluxes, wavelengths = TXTReader.read(self.path_to_spectrum)
        #     binned_spectral_flux_density = spectres.spectres(
        #         self._phringe._instrument.wavelength_bin_centers.numpy(),
        #         wavelengths.numpy(),
        #         fluxes.numpy(),
        #         fill=0,
        #         verbose=False
        #     ) * self._solid_angle
        #     return torch.asarray(binned_spectral_flux_density, dtype=torch.float32, device=self._phringe._device)
        else:
            binned_spectral_flux_density = torch.asarray(
                get_blackbody_spectrum_standard_units(
                    self.temperature,
                    self._phringe._instrument.wavelength_bin_centers
                )
                , dtype=torch.float32, device=self._phringe._device) * self._solid_angle
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

        sky_coordinates_at_time_step = get_meshgrid(2 * (1.2 * angular_radius), self._phringe._grid_size,
                                                    device=self._phringe._device)

        return torch.stack((sky_coordinates_at_time_step[0], sky_coordinates_at_time_step[1]))

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
        host_star_distance = self.host_star_distance if self.host_star_distance is not None else self._phringe._scene.star.distance
        separation_from_star_x, separation_from_star_y = self._get_x_y_separation_from_star(time_step)
        angular_separation_from_star_x = separation_from_star_x / host_star_distance
        angular_separation_from_star_y = separation_from_star_y / host_star_distance
        return (angular_separation_from_star_x, angular_separation_from_star_y)

    def _get_x_y_separation_from_star(self, time_step: float, ) -> Tuple:
        """Return the separation of the planet from the star in x- and y-direction. If the planet orbital motion is
        considered, calculate the new position for each time step.

        :param time_step: The time step
        :param has_planet_orbital_motion: Whether the planet orbital motion is to be considered
        :param star_mass: The mass of the star
        :return: A tuple containing the x- and y- coordinates
        """
        host_star_mass = self.host_star_mass if self.host_star_mass is not None else self._phringe._scene.star.mass
        star = Body(parent=None, k=G * (host_star_mass + self.mass) * u.kg, name='Star')
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

            x, y = _convert_orbital_elements_to_sky_position(a, e, i, Omega, omega, M)
        return x, y

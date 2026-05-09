from functools import cached_property
from typing import Any, Tuple, Union

import numpy as np
import torch
from astropy import units as u
from astropy.constants.codata2018 import G
from astropy.units import Quantity
from phringe.io.input_spectrum import SEDLoader
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_index_of_closest_value, get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_si_units


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
    sed_loader: InputSpectrum, optional
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
    sed_loader: Union[SEDLoader, None]
    grid_position: Tuple = None
    host_star_distance: Union[str, float, Quantity] = None
    host_star_mass: Union[str, float, Quantity] = None
    _max_ang_sep_from_star_x: Any = None
    _max_ang_sep_from_star_y: Any = None
    _simulation_time_steps: Any = None

    @field_validator('argument_of_periapsis')
    def _validate_argument_of_periapsis(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the argument of periapsis input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Argument of periapsis in units of degrees.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('inclination')
    def _validate_inclination(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the inclination input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Inclination in units of degrees.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('mass')
    def _validate_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the mass input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Mass in units of kilograms.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,))

    @field_validator('raan')
    def _validate_raan(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the right ascension of the ascending node input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Right ascension of the ascending node in units of degrees.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('radius')
    def _validate_radius(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the radius input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Radius in units of meters.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('semi_major_axis')
    def _validate_semi_major_axis(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the semi-major axis input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Semi-major axis in units of meters.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('temperature')
    def _validate_temperature(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the temperature input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Temperature in units of Kelvin.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.K,))

    @field_validator('true_anomaly')
    def _validate_true_anomaly(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the true anomaly input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            True anomaly in units of degrees.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('host_star_distance')
    def _validate_host_star_distance(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the host star distance input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Host star distance in units of meters.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('host_star_mass')
    def _validate_host_star_mass(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the host star mass input.

        Parameters
        ----------
        value : Any
            Value given as input.
        info : ValidationInfo
            Validation information for the field.

        Returns
        -------
        float
            Host star mass in units of kilograms.
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.kg,))

    @cached_property
    def _proj_ang_pos(self) -> Tensor:
        """Return the projected angular position of the planet as a tensor of shape 2 x 1 or 2 x n_time_steps.

        Returns
        -------
        torch.Tensor
            Tensor containing the projected angular position of the planet.
        """
        host_star_distance = (
            self.host_star_distance
            if self.host_star_distance is not None
            else self._phringe._scene.star.distance
        )
        return self._get_proj_sky_pos() / host_star_distance

    @property
    def n_grid_points(self) -> int:
        return 1

    @property
    def sky_brightness_distribution(self) -> Tensor:
        n_wavelengths = len(self._phringe._instrument.wavelength_bin_centers)
        n_grid = self._phringe._grid_size
        n_time_steps = len(self.sky_coordinates[1])
        device = self._phringe._device

        sky_brightness_distribution = torch.zeros((n_wavelengths, n_time_steps, n_grid, n_grid), device=device)
        ang_proj_sky_pos = self._proj_ang_pos

        ix = get_index_of_closest_value(
            self.sky_coordinates[0, 0, :, 0, :],
            ang_proj_sky_pos[0]
        ) if self.grid_position is None else self.grid_position[0]

        iy = get_index_of_closest_value(
            self.sky_coordinates[1, 0, :, :, 0],
            ang_proj_sky_pos[1]
        ) if self.grid_position is None else self.grid_position[1]

        it = torch.arange(n_time_steps, device=device)

        sky_brightness_distribution[:, it, ix, iy] = self.spectral_energy_distribution[:, None, 0, 0]

        return sky_brightness_distribution

    @property
    def sky_coordinates(self) -> Tensor:
        n_grid = self._phringe._grid_size

        ang_proj_pos = self._proj_ang_pos
        ang_proj_pos_x = ang_proj_pos[0]
        ang_proj_pos_y = ang_proj_pos[1]

        angular_radius = torch.sqrt(ang_proj_pos_x ** 2 + ang_proj_pos_y ** 2)

        angular_sky_coordinates = get_meshgrid(
            2 * (1.2 * angular_radius),
            n_grid,
            device=self._phringe._device
        )

        # Broadcast to time dimension
        return angular_sky_coordinates[:, None, :, :, :]

    @property
    def solid_angle(self) -> Union[float, Tensor]:
        host_star_distance = (
            self.host_star_distance
            if self.host_star_distance is not None
            else self._phringe._scene.star.distance
        )
        return torch.pi * (self.radius / host_star_distance) ** 2

    @property
    def spectral_energy_distribution(self) -> Tensor:
        if self.sed_loader is not None:
            spectral_energy_distribution = self.sed_loader.get_spectral_energy_distribution(
                self._phringe._instrument.wavelength_bin_centers,
                self.solid_angle,
                self._phringe._device
            )

        else:
            spectral_energy_distribution = torch.asarray(
                get_blackbody_spectrum_si_units(
                    self.temperature,
                    self._phringe._instrument.wavelength_bin_centers
                )
                , dtype=torch.float32,
                device=self._phringe._device
            ) * self.solid_angle

        # Broadcast to wavelength dimension
        return spectral_energy_distribution[:, None, None]

    def _get_proj_sky_pos(self) -> Tensor:
        """Return the projected x- and y-position of the planet on the sky as a tensor of shape 2 x 1 or 2 x n_time_steps.

        Returns
        -------
        torch.Tensor
            Tensor containing the projected x- and y-position of the planets on the sky.
        """
        host_star_mass = (
            self.host_star_mass
            if self.host_star_mass is not None
            else self._phringe._scene.star.mass
        )
        star = Body(parent=None, k=G * (host_star_mass + self.mass) * u.kg, name='Star')

        orbit = Orbit.from_classical(
            star,
            a=self.semi_major_axis * u.m,
            ecc=u.Quantity(self.eccentricity),
            inc=self.inclination * u.rad,
            raan=self.raan * u.rad,
            argp=self.argument_of_periapsis * u.rad,
            nu=self.true_anomaly * u.rad
        )

        if self.has_orbital_motion:
            propagation_time_steps = self._phringe.simulation_time_steps.cpu().numpy()
        else:
            propagation_time_steps = [0]

        states = [orbit.propagate(t * u.s) for t in propagation_time_steps]
        rr = np.array([state.r.to(u.m).value for state in states])

        pos_x = torch.tensor(rr[:, 1], device=self._phringe._device)
        pos_y = torch.tensor(rr[:, 0], device=self._phringe._device)

        return torch.stack([pos_x, pos_y], dim=0)

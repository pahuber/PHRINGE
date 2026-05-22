import math
from functools import cached_property
from typing import Any, Tuple, Union

import torch
from astropy import units as u
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.io.sed_loader import SEDLoader
from phringe.io.validation import validate_quantity_units
from phringe.util.spectrum import get_blackbody_spectrum_si_units


class Planet(BaseSource):
    """Class representation of a planet.

    Parameters
    ----------
    propagate_orbit: bool
        Whether the planet's orbit is propagated in time. If not, it is assumed to be static.
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
    sed_loader: SEDLoader, optional
        The object to load custom SEDs. If None, a blackbody spectrum is generated.
    grid_position: Tuple[int, int], optional
        The grid position of the planet in the sky. If None, the position is calculated from its orbital elements.
    """
    name: str
    propagate_orbit: bool
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

    @cached_property
    def _proj_ang_pos(self) -> Tensor:
        """Return the projected angular position of the planet as a tensor of shape 2 x 1 or 2 x n_time_steps.

        Returns
        -------
        torch.Tensor
            Tensor containing the projected angular position of the planet.
        """
        host_star_distance = (
            self._phringe._scene.star.distance
            if self._phringe._scene.star is not None
            else self._phringe._observation.host_star_distance
        )
        return self._get_proj_sky_pos() / host_star_distance

    @property
    def n_grid_points(self) -> int:
        return 1

    @property
    def sky_brightness_distribution(self) -> Tensor:
        return self.spectral_energy_distribution[:, None, :, :]

    @property
    def sky_coordinates(self) -> Tensor:
        ang_proj_pos = self._proj_ang_pos
        ang_proj_pos_x = ang_proj_pos[0]
        ang_proj_pos_y = ang_proj_pos[1]

        return torch.stack([ang_proj_pos_y, ang_proj_pos_x], dim=0)[:, None, :, None, None]

    @property
    def solid_angle(self) -> Union[float, Tensor]:
        host_star_distance = (
            self._phringe._scene.star.distance
            if self._phringe._scene.star is not None
            else self._phringe._observation.host_star_distance
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

    @staticmethod
    def _solve_kepler(M: Tensor, e: Tensor, n_iter: int = 8) -> Tensor:
        E = M.clone()

        for _ in range(n_iter):
            E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))

        return E

    def _get_proj_sky_pos(self) -> Tensor:
        """Return the projected x- and y-position of the planet on the sky as a tensor of shape 2 x 1 or 2 x n_time_steps.

        Returns
        -------
        torch.Tensor
            Tensor containing the projected x- and y-position of the planets on the sky.
        """
        times = self._phringe.simulation_time_steps.to(device=self._phringe._device)

        if not self.propagate_orbit:
            times = torch.zeros_like(times)

        y, x = self._propagate_kepler_orbit(times)

        return torch.stack([x, y], dim=0)

    def _propagate_kepler_orbit(self, times: Tensor) -> Tuple[Tensor, Tensor]:
        """Propagate the planet's orbit to the given times.

        Parameters
        ----------
        times : torch.Tensor
            Times to propagate the orbit to.

        Returns
        -------

        """
        device = self._phringe._device
        dtype = torch.float32

        host_star_mass = (
            self._phringe._scene.star.mass
            if self._phringe._scene.star is not None
            else self._phringe._observation.host_star_mass
        )

        a = torch.tensor(self.semi_major_axis, dtype=dtype, device=device)
        e = torch.tensor(self.eccentricity, dtype=dtype, device=device)
        inc = torch.tensor(self.inclination, dtype=dtype, device=device)
        raan = torch.tensor(self.raan, dtype=dtype, device=device)
        argp = torch.tensor(self.argument_of_periapsis, dtype=dtype, device=device)
        f0 = torch.tensor(self.true_anomaly, dtype=dtype, device=device)

        mu = torch.tensor(
            6.67430e-11 * (host_star_mass + self.mass),
            dtype=dtype,
            device=device,
        )

        # Initial true anomaly -> eccentric anomaly
        E0 = 2 * torch.atan2(
            torch.sqrt(1 - e) * torch.sin(f0 / 2),
            torch.sqrt(1 + e) * torch.cos(f0 / 2),
        )

        M0 = E0 - e * torch.sin(E0)

        # Mean motion
        n = torch.sqrt(mu / a ** 3)

        M = M0 + n * times
        M = torch.remainder(M, 2 * math.pi)

        E = self._solve_kepler(M, e)

        # Position in orbital plane
        x_orb = a * (torch.cos(E) - e)
        y_orb = a * torch.sqrt(1 - e ** 2) * torch.sin(E)

        cos_O = torch.cos(raan)
        sin_O = torch.sin(raan)
        cos_i = torch.cos(inc)
        sin_i = torch.sin(inc)
        cos_w = torch.cos(argp)
        sin_w = torch.sin(argp)

        # Rotate from orbital plane to sky frame
        x = (
                (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb
                + (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb
        )

        y = (
                (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb
                + (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb
        )

        return x, y

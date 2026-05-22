from typing import Union

import torch
from astropy import units as u
from astropy.units import Quantity
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.util.coordinates import get_ecliptic_coordinates
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_si_units


class LocalZodi(BaseSource):
    """Class representation of a local zodi."""
    _solar_ecliptic_latitude: Union[str, float, Quantity] = None

    @property
    def n_grid_points(self) -> int:
        return self._phringe._grid_size ** 2

    @property
    def sky_brightness_distribution(self) -> Tensor:
        sky_brightness_distribution = self.spectral_energy_distribution

        # Broadcast to time dimension
        return sky_brightness_distribution[:, None, :, :]

    @property
    def sky_coordinates(self) -> Tensor:
        sky_coordinates = get_meshgrid(
            self._phringe._instrument._field_of_view,
            self._phringe._grid_size,
            self._phringe._device,
        )

        # Broadcast to time dimension
        return sky_coordinates[:, :, None, :, :]

    @property
    def solid_angle(self) -> Union[float, Tensor]:
        return self._phringe._instrument._field_of_view ** 2

    @property
    def spectral_energy_distribution(self) -> Tensor:
        tau = 4e-8
        a = 0.22
        wavelengths = self._phringe._instrument.wavelength_bin_centers
        #
        host_star_right_ascension = (
            self._phringe._scene.star.right_ascension
            if self._phringe._scene.star is not None
            else self._phringe._observation.host_star_right_ascension
        )
        host_star_declination = (
            self._phringe._scene.star.declination
            if self._phringe._scene.star is not None
            else self._phringe._observation.host_star_declination
        )
        solar_ecliptic_latitude = (
            self._solar_ecliptic_latitude
            if self._solar_ecliptic_latitude is not None
            else self._phringe._observation.solar_ecliptic_latitude
        )

        ecliptic_latitude, relative_ecliptic_longitude = get_ecliptic_coordinates(
            host_star_right_ascension,
            host_star_declination,
            solar_ecliptic_latitude
        )

        spectral_energy_distribution = (
                tau * (get_blackbody_spectrum_si_units(265, wavelengths) * self.solid_angle
                       + a * get_blackbody_spectrum_si_units(5778, wavelengths) * self.solid_angle
                       * ((1 * u.Rsun).to(u.au) / (1.5 * u.au)).value ** 2)
                * ((torch.pi / torch.arccos(torch.cos(torch.tensor(relative_ecliptic_longitude))
                                            * torch.cos(torch.tensor(ecliptic_latitude))))
                   / (torch.sin(torch.tensor(ecliptic_latitude)) ** 2
                      + 0.6 * (wavelengths / (11e-6)) ** (-0.4)
                      * torch.cos(torch.tensor(ecliptic_latitude)) ** 2
                      )
                   ) ** 0.5)

        # Broadcast to grid dimension
        return spectral_energy_distribution[:, None, None]

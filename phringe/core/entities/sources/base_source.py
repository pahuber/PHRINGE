from abc import abstractmethod, ABC
from typing import Any, Union

from pydantic import BaseModel
from torch import Tensor

from phringe.core.cached_attributes_entity import CachedAttributesEntity


class CachedAttributesSource(ABC, BaseModel, CachedAttributesEntity):
    """Class representation of a photon source1.

    :param mean_spectral_flux_density: An array containing the mean spectral flux density of the photon source1 for each
        wavelength in units of ph/(s * um * m**2). If the mean spectral flux density is constant over time, then the
        time axis is omitted
    :param sky_brightness_distribution: An array containing for each time and wavelength a grid with the sky
        brightness distribution of the photon source1 in units of ph/(s * um * m**2). If the sky brightness distribution
        is constant over time, then the time axis is omitted
    :param sky_coordinates: An array containing the sky coordinates for each time and wavelength in units of radians.
        If the sky coordinates are constant over time and/or wavelength, the time/wavelength axes are omitted
    """
    name: str = None
    __spectral_energy_density: Any = None
    __sky_brightness_distribution: Any = None
    __sky_coordinates: Any = None
    _solid_angle: Any = None
    _grid_size: int = None
    _instrument: Any = None

    @property
    @abstractmethod
    def _spectral_energy_distribution(self) -> Union[Tensor, None]:
        """Return the mean spectral flux density of the source1 object for each wavelength.

        :param wavelength_steps: The wavelength steps
        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        :return: The mean spectral flux density
        """
        pass

    @property
    @abstractmethod
    def _sky_brightness_distribution(self) -> Union[Tensor, None]:
        """Calculate and return the sky brightness distribution of the source1 object for each (time and) wavelength as
        an array of shape N_wavelengths x N_pix x N_pix or N_time_steps x N_wavelengths x N_pix x N_pix (e.g. when
        accounting for planetary orbital motion).

        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        :return: The sky brightness distribution
        """
        pass

    @property
    @abstractmethod
    def _sky_coordinates(self) -> Union[Tensor, None]:
        """Calculate and return the sky coordinates of the source1 for a given time. For moving sources, such as planets,
         the sky coordinates might change over time to ensure optimal sampling, e.g. for a planet that moves in very
         close to the star). The sky coordinates for the different sources are of the following shapes:
            - star: 2 x N_pix x N_pix
            - planet: 2 x N_pix x N_pix (no motion) or 2 x N_time_steps x N_pix x N_pix (with motion)
            - local and exozodi: 2 x N_wavelength x N_pix x N_pix (N_wavelength, since they fill the whole FoV, which is
              wavelength-dependent).

        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        :return: A coordinates object containing the x- and y-sky coordinate maps
        """
        pass

    @property
    @abstractmethod
    def solid_angle(self) -> Union[float, Tensor]:
        """Calculate and return the solid angle of the source1 object.

        :param kwargs: Additional keyword arguments
        :return: The solid angle
        """
        pass

    @abstractmethod
    def _get_sky_brightness_distribution(self):
        pass

    @abstractmethod
    def _get_sky_coordinates(self):
        pass

    @abstractmethod
    def _get_spectral_energy_distribution(self):
        pass

    @abstractmethod
    def _get_solid_angle(self):
        pass

    #
    # def prepare(self):
    #     """Prepare the photon source1 for the simulation. This method is called before the simulation starts and can be
    #     used to pre-calculate values that are constant over time and/or wavelength.
    #
    #     :param wavelength_bin_centers: The wavelength steps
    #     :param grid_size: The grid size
    #     :param kwargs: Additional keyword arguments
    #     """
    #     self.solid_angle = self.solid_angle()
    #     self.spectral_flux_density = self.spectral_energy_distribution()
    #     self.sky_coordinates = self.sky_coordinates()
    #     self.sky_brightness_distribution = self.sky_brightness_distribution()

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
from pydantic import BaseModel
from torch import Tensor


class BasePhotonSource(ABC, BaseModel):
    """Class representation of a photon source.

    :param mean_spectral_flux_density: An array containing the mean spectral flux density of the photon source for each
        wavelength in units of ph/(s * um * m**2). If the mean spectral flux density is constant over time, then the
        time axis is omitted
    :param sky_brightness_distribution: An array containing for each time and wavelength a grid with the sky
        brightness distribution of the photon source in units of ph/(s * um * m**2). If the sky brightness distribution
        is constant over time, then the time axis is omitted
    :param sky_coordinates: An array containing the sky coordinates for each time and wavelength in units of radians.
        If the sky coordinates are constant over time and/or wavelength, the time/wavelength axes are omitted
    """
    name: str = None
    spectral_flux_density: Any = None
    sky_brightness_distribution: Any = None
    sky_coordinates: Any = None
    solid_angle: Any = None

    @abstractmethod
    def _calculate_spectral_flux_density(self, wavelength_steps: np.ndarray, grid_size: int,
                                         **kwargs) -> np.ndarray:
        """Return the mean spectral flux density of the source object for each wavelength.

        :param wavelength_steps: The wavelength steps
        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        :return: The mean spectral flux density
        """
        pass

    @abstractmethod
    def _calculate_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        """Calculate and return the sky brightness distribution of the source object for each (time and) wavelength.

        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        :return: The sky brightness distribution
        """
        pass

    @abstractmethod
    def _calculate_sky_coordinates(self, grid_size: int, **kwargs) -> np.ndarray:
        """Calculate and return the sky coordinates of the source for a given time. For moving sources, such as planets,
         the sky coordinates might change over time.

        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        :return: A coordinates object containing the x- and y-sky coordinate maps
        """
        pass

    @abstractmethod
    def _calculate_solid_angle(self, **kwargs) -> Union[float, Tensor]:
        """Calculate and return the solid angle of the source object.

        :param kwargs: Additional keyword arguments
        :return: The solid angle
        """
        pass

    def prepare(self, wavelength_bin_centers, grid_size, **kwargs):
        """Prepare the photon source for the simulation. This method is called before the simulation starts and can be
        used to pre-calculate values that are constant over time and/or wavelength.

        :param wavelength_bin_centers: The wavelength steps
        :param grid_size: The grid size
        :param kwargs: Additional keyword arguments
        """
        self.solid_angle = self._calculate_solid_angle(**kwargs)
        self.spectral_flux_density = self._calculate_spectral_flux_density(
            wavelength_bin_centers,
            grid_size,
            **kwargs
        )
        self.sky_coordinates = self._calculate_sky_coordinates(grid_size, **kwargs)
        self.sky_brightness_distribution = self._calculate_sky_brightness_distribution(grid_size, **kwargs)

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.sygn.core.util.helpers import Coordinates


class BasePhotonSource(ABC, BaseModel):
    """Class representation of a photon source.

    :param sky_brightness_distribution: An array containing for each time and wavelength a grid with the sky
        brightness distribution of the photon source in units of ph/(s * um * m**2). If the sky brightness distribution
        is constant over time, then the time axis is omitted
    :param sky_coordinates: An array containing the sky coordinates for each time and wavelength in units of radians.
        If the sky coordinates are constant over time and/or wavelength, the time/wavelength axes are omitted
    """

    mean_spectral_flux_density: Any = None
    sky_brightness_distribution: Any = None
    sky_coordinates: Any = None

    @abstractmethod
    def _calculate_mean_spectral_flux_density(self,
                                              wavelength_steps: np.ndarray,
                                              **kwargs) -> np.ndarray:
        """Return the mean spectral flux density of the source object for each wavelength.

        :return: The mean spectral flux density
        """
        pass

    @abstractmethod
    def _calculate_sky_brightness_distribution(self, grid_size: int, **kwargs) -> np.ndarray:
        """Calculate and return the sky brightness distribution of the source object for each (time and) wavelength.

        :param context: The context
        :return: The sky brightness distribution
        """
        pass

    @abstractmethod
    def _calculate_sky_coordinates(self, grid_size: int, **kwargs) -> Coordinates:
        """Calculate and return the sky coordinates of the source for a given time. For moving sources, such as planets,
         the sky coordinates might change over time.

        :return: A tuple containing the x- and y-sky coordinate maps
        """
        pass

    def prepare(self,
                wavelength_steps,
                grid_size,
                **kwargs):
        """Prepare the photon source for the simulation. This method is called before the simulation starts and can be
        used to pre-calculate values that are constant over time and/or wavelength.
        """
        self.mean_spectral_flux_density = self._calculate_mean_spectral_flux_density(wavelength_steps, **kwargs)
        self.sky_coordinates = self._calculate_sky_coordinates(grid_size, **kwargs)
        self.sky_brightness_distribution = self._calculate_sky_brightness_distribution(grid_size, **kwargs)
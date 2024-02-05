from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
from pydantic import BaseModel


class BasePhotonSource(ABC, BaseModel):
    """Class representation of a photon source.

    :param mean_spectral_flux_density: An array containing for each wavelength the mean spectral flux density in units
        of ph/(s * um * m**2)
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
    def _calculate_mean_spectral_flux_density(self, context: Context) -> np.ndarray:
        """Calculate and return the mean spectral flux density of the source object for each wavelength.

        :param context: The context
        :return: The mean spectral flux density
        """
        pass

    @abstractmethod
    def _calculate_sky_brightness_distribution(self, context: Context) -> np.ndarray:
        """Calculate and return the sky brightness distribution of the source object for each (time and) wavelength.

        :param context: The context
        :return: The sky brightness distribution
        """
        pass

    @abstractmethod
    def _calculate_sky_coordinates(
            self, context: Context
    ) -> Union[np.ndarray, Coordinates]:
        """Calculate and return the sky coordinates of the source (for each time).

        :param context: The context
        :return: A tuple containing the x- and y-sky coordinate maps
        """
        pass

    @abstractmethod
    def get_sky_brightness_distribution(
            self, index_time: int, index_wavelength: int
    ) -> np.ndarray:
        """Return the sky brightness distribution for the respective time and wavelength index.

        :param index_time: The time index
        :param index_wavelength: The wavelength index
        :return: The sky brightness distribution
        """
        pass

    @abstractmethod
    def get_sky_coordinates(
            self, index_time: int, index_wavelength: int
    ) -> Coordinates:
        """Return the sky coordinates for the respective time and wavelength index.

        :param index_time: The time index
        :param index_wavelength: The wavelength index
        :return: The sky coordinates
        """
        pass

    def setup(self, context: Context):
        """Set up the main properties of the photon source. Rather than calling this method on initialization of the
        class instance, it has to be called explicitly after initiating the photon source object. This ensures a
        flexibility in adapting source properties (e.g. temperature) on the fly without having to load a separate
        configuration file for each adaptation, as the setup also requires information that is not source specific.

        :param context: The context
        """
        self.mean_spectral_flux_density = self._calculate_mean_spectral_flux_density(
            context
        )
        self.sky_coordinates = self._calculate_sky_coordinates(context)
        self.sky_brightness_distribution = self._calculate_sky_brightness_distribution(
            context
        )

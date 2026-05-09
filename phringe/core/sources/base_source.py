from abc import abstractmethod, ABC
from typing import Union

from torch import Tensor

from phringe.core.base_entity import BaseEntity


class BaseSource(ABC, BaseEntity):
    """Class representation of a photon source in the scene.

    Parameters
    ----------

    """

    @property
    @abstractmethod
    def n_grid_points(self) -> int:
        """Return the number of grid points (pixels) covered by the source.

        Returns
        -------
        int
            The number of grid points.
        """

    @property
    @abstractmethod
    def spectral_energy_distribution(self) -> Tensor:
        """Return the spectral energy distribution of the source of shape n_wavelengths x n_grid x n_grid.

        Returns
        -------
        torch.Tensor
            The spectral energy distribution as a 1D array of shape n_wavelengths.
        """
        pass

    @property
    @abstractmethod
    def sky_brightness_distribution(self) -> Tensor:
        """Return the angular sky brightness distribution of the source of shape n_wavelengths x n_time_steps x n_grid x n_grid.

        Returns
        -------
        torch.Tensor
            The sky brightness distribution as a 4D array of shape n_wavelengths x n_time_steps x n_grid x n_grid
        """
        pass

    @property
    @abstractmethod
    def sky_coordinates(self) -> Tensor:
        """Return the angular sky coordinates of the source of shape 2 x n_wavelengths x n_time_steps x n_grid x n_grid.

        Returns
        -------
        torch.Tensor
            The angular sky coordinates as a 4D array of shape 2 x n_wavelengths x n_time_steps x n_grid x n_grid
        """
        pass

    @property
    @abstractmethod
    def solid_angle(self) -> Union[float, Tensor]:
        """Return the solid angle of the source of shape n_wavelengths.

        Returns
        -------
        torch.Tensor
            They solid angle of the source as a 1D array of shape n_wavelengths.
        """
        pass

from typing import Tuple

import numpy as np
import torch
from astropy.units import Quantity
from torch import Tensor


def get_meshgrid(full_extent: float, grid_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return a tuple of numpy arrays corresponding to a meshgrid.

    :param full_extent: Full extent in one dimension
    :param grid_size: Grid size
    :return: Tuple of numpy arrays
    """
    linspace = torch.linspace(-full_extent / 2, full_extent / 2, grid_size, device=device)
    return torch.meshgrid((linspace, linspace), indexing='ij')


def get_radial_map(full_extent: Quantity, grid_size: int, device=torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return a radial map over the full extent given.

    :param full_extent: The full extent
    :param grid_size: The grid size
    :return: THe radial map
    """
    meshgrid = get_meshgrid(full_extent, grid_size, device)
    return torch.sqrt(meshgrid[0] ** 2 + meshgrid[1] ** 2)


def get_index_of_closest_value_numpy(array: np.ndarray, value: Quantity):
    """Return the index of a value in an array closest to the provided value.

    :param array: The array to search in
    :param value: The value to check
    :return: The index of the closest value
    """
    return np.abs(array - value).argmin()


def get_index_of_closest_value(array: Tensor, value: Tensor):
    """Return the index of a value in an array closest to the provided value.

    :param array: The array to search in
    :param value: The value to check
    :return: The index of the closest value
    """
    return torch.abs(array - value).argmin()

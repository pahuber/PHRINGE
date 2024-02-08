from typing import Tuple, Any

import numpy as np
from astropy.units import Quantity


def get_meshgrid(full_extent: Quantity, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a tuple of numpy arrays corresponding to a meshgrid.

    :param full_extent: Full extent in one dimension
    :param grid_size: Grid size
    :return: Tuple of numpy arrays
    """
    linspace = np.linspace(-full_extent.value / 2, full_extent.value / 2, grid_size)
    return np.meshgrid(linspace, linspace) * full_extent.unit


def get_radial_map(full_extent: Quantity, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a radial map over the full extent given.

    :param full_extent: The full extent
    :param grid_size: The grid size
    :return: THe radial map
    """
    meshgrid = get_meshgrid(full_extent, grid_size)
    return np.sqrt(meshgrid[0] ** 2 + meshgrid[1] ** 2)


def get_index_of_closest_value(array: np.ndarray, value: Quantity):
    """Return the index of a value in an array closest to the provided value.

    :param array: The array to search in
    :param value: The value to check
    :return: The index of the closest value
    """
    return np.abs(array - value).argmin()


def get_number_of_instances_in_list(list: list, instance_type: Any) -> int:
    """Return the number of objects of a given instance in a list.

    :param list: The list
    :param instance_type: The type of instance
    :return: The number of objects
    """
    return len([value for value in list if isinstance(value, instance_type)])


def get_indices_of_maximum_of_2d_array(array: np.ndarray) -> Tuple[int, int]:
    """Return the indices of the maximum of a 2D array.

    :param array: The array
    :return: The indices of the maximum
    """
    index = np.where(array == np.max(array))
    return index[0][0], index[1][0]

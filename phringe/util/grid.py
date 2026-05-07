from typing import Tuple, Union

import numpy as np
import torch
from astropy.units import Quantity
from torch import Tensor


#
# def get_meshgrid(full_extent: float, grid_size: int, device: torch.device) -> Tensor:
#     """Return a tuple of numpy arrays corresponding to a meshgrid.
#
#     :param full_extent: Full extent in one dimension
#     :param grid_size: Grid size
#     :return: Tuple of numpy arrays
#     """
#     extent = torch.linspace(-full_extent / 2, full_extent / 2, grid_size, device=device)
#     meshgrid = torch.meshgrid((extent, extent), indexing='ij')
#     return torch.stack([meshgrid[1], meshgrid[0]])


def get_meshgrid(
        full_extent: Union[float, torch.Tensor],
        grid_size: int,
        device: torch.device
) -> Tensor:
    full_extent = torch.as_tensor(full_extent, device=device)

    base_extent = torch.linspace(
        -0.5,
        0.5,
        grid_size,
        device=device
    )

    if full_extent.ndim == 0:
        extent = base_extent * full_extent
        meshgrid = torch.meshgrid((extent, extent), indexing="ij")
        return torch.stack([meshgrid[1], meshgrid[0]])

    extent = full_extent[:, None] * base_extent[None, :]  # (T, G)

    x = extent[:, None, :].expand(-1, grid_size, -1)  # (T, G, G)
    y = extent[:, :, None].expand(-1, -1, grid_size)  # (T, G, G)

    return torch.stack([x, y], dim=0)  # (2, T, G, G)


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


# def get_index_of_closest_value(array: Tensor, value: Tensor):
#     """Return the index of a value in an array closest to the provided value.
#
#     :param array: The array to search in
#     :param value: The value to check
#     :return: The index of the closest value
#     """
#     return torch.abs(array - value).argmin()
#
# def get_index_of_closest_value(array: torch.Tensor,
#                                values: torch.Tensor):
#     """
#     array  : shape (N,)
#     values : shape (M,)
#
#     returns:
#         indices : shape (M,)
#     """
#
#     distances = torch.abs(array[None, :] - values[:, None])
#
#     return distances.argmin(dim=1)

def get_index_of_closest_value(
        array: torch.Tensor,
        values: torch.Tensor
):
    """
    array  : shape (M, N)
    values : shape (M,)

    returns:
        indices : shape (M,)
    """

    distances = torch.abs(array - values[:, None])

    return distances.argmin(dim=1)

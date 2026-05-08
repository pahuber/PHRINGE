from typing import Tuple, Union

import numpy as np
import torch
from astropy.units import Quantity
from torch import Tensor


def get_index_of_closest_value(array: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    array  : shape (M, N)
    values : shape (M,)

    returns:
        indices : shape (M,)
    """
    distances = torch.abs(array - values[:, None])

    return distances.argmin(dim=1)


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

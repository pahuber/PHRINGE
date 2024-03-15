import numpy as np
import torch
from astropy.units import Quantity
from torch import cos, sin, pi


def get_2d_rotation_matrix(time: Quantity, rotation_period: Quantity) -> np.ndarray:
    """Return the matrix corresponding to a 2D rotation.

    :param time: Time variable in seconds
    :param rotation_period: Rotation period for a full rotation in seconds
    :return: An array containing the matrix
    """
    argument = torch.asarray(2 * pi / rotation_period * time, dtype=torch.float32).unsqueeze(0).squeeze()
    return torch.stack([
        cos(argument), -sin(argument),
        sin(argument), cos(argument)
    ], dim=1).view(-1, 2, 2)

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import astropy.units
import numpy as np
import torch
from astropy import units as u
from astropy.units import Quantity
from pydantic import BaseModel
from torch import tensor, Tensor

from phringe.util.helpers import Coordinates
from phringe.util.matrix import get_2d_rotation_matrix


class ArrayEnum(Enum):
    """Enum representing the different array types.
    """
    EMMA_X_CIRCULAR_ROTATION = 'emma-x-circular-rotation'
    EMMA_X_DOUBLE_STRETCH = 'emma-x-double-stretch'
    EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION = 'equilateral-triangle-circular-rotation'
    REGULAR_PENTAGON_CIRCULAR_ROTATION = 'regular-pentagon-circular-rotation'


class Array(ABC, BaseModel):
    """Class representation of a collector array.

    :param nulling_baseline_length: The length of the nulling baseline
    :param type: The type of the array configuration
    """
    # nulling_baseline_length: Any = None
    type: Any = None
    collector_coordinates: Any = None

    @abstractmethod
    def get_collector_coordinates(
            self,
            time_steps: Tensor,
            nulling_baseline: float,
            modulation_period: float,
            baseline_ratio: int
    ) -> Tensor:
        """Return time-dependent x- and y-coordinates of the collectors as an array of shape 2 x N_collectors x
        N_time_steps.

        :param time_steps: The time steps for which the collector positions are calculated
        :param modulation_period: The modulation period of the array
        :param baseline_ratio: The baseline ratio of the array
        :return: The array containing the coordinates of the collectors for each time step
        """
        pass


class EmmaXCircularRotation(Array):
    """Class representation of the Emma-X array configuration with circular rotation of the array.
    """
    type: Any = ArrayEnum.EMMA_X_CIRCULAR_ROTATION

    def get_collector_coordinates(
            self,
            time_steps: Tensor,
            nulling_baseline: float,
            modulation_period: float,
            baseline_ratio: int
    ) -> Tensor:
        rotation_matrix = get_2d_rotation_matrix(time_steps, modulation_period)
        emma_x_static = nulling_baseline / 2 * torch.asarray(
            [[baseline_ratio, baseline_ratio, -baseline_ratio, -baseline_ratio], [1, -1, -1, 1]], dtype=torch.float32)
        collector_positions = torch.einsum('ijl,jk->ikl', rotation_matrix, emma_x_static)
        return collector_positions.swapaxes(0, 2)


class EmmaXDoubleStretch(Array):
    """Class representation of the Emma-X array configuration with double stretching of the array.
    """
    type: Any = ArrayEnum.EMMA_X_DOUBLE_STRETCH

    def get_collector_coordinates(self,
                                  time_steps: np.ndarray,
                                  modulation_period: Quantity,
                                  baseline_ratio: int) -> np.ndarray:
        emma_x_static = self.nulling_baseline_length / 2 * np.array(
            [[baseline_ratio, baseline_ratio, -baseline_ratio, -baseline_ratio], [1, -1, -1, 1]])
        # TODO: fix calculations
        collector_positions = emma_x_static * (
                1 + (2 * self.nulling_baseline_length) / self.nulling_baseline_length * np.sin(
            2 * np.pi * u.rad / modulation_period * time_steps))
        return Coordinates(collector_positions[0], collector_positions[1])


class EquilateralTriangleCircularRotation(Array):
    """Class representation of an equilateral triangle configuration with circular rotation of the array.
    """
    type: Any = ArrayEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION

    def get_collector_coordinates(
            self,
            time_steps: Tensor,
            nulling_baseline: float,
            modulation_period: float,
            baseline_ratio: int
    ) -> Tensor:
        height = torch.sqrt(torch.tensor(3)) / 2 * nulling_baseline
        height_to_center = height / 3
        rotation_matrix = get_2d_rotation_matrix(time_steps, modulation_period)

        equilateral_triangle_static = torch.asarray(
            [[0, nulling_baseline / 2, -nulling_baseline / 2],
             [height - height_to_center, -height_to_center, -height_to_center]], dtype=torch.float32)
        collector_positions = torch.einsum('ijl,jk->ikl', rotation_matrix,
                                           equilateral_triangle_static)
        return collector_positions.swapaxes(0, 2)


class RegularPentagonCircularRotation(Array):
    """Class representation of a regular pentagon configuration with circular rotation of the array.
    """
    type: Any = ArrayEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION

    def _get_x_position(self, angle, nulling_baseline) -> astropy.units.Quantity:
        """Return the x position.

        :param angle: The angle at which the collector is located
        :return: The x position
        """
        return 0.851 * nulling_baseline * torch.cos(angle)

    def _get_y_position(self, angle, nulling_baseline) -> astropy.units.Quantity:
        """Return the y position.

        :param angle: The angle at which the collector is located
        :return: The y position
        """
        return 0.851 * nulling_baseline * torch.sin(angle)

    def get_collector_coordinates(
            self,
            time_steps: Tensor,
            nulling_baseline: float,
            modulation_period: float,
            baseline_ratio: int
    ) -> Tensor:
        angles = [tensor(0), tensor(2 * torch.pi / 5), tensor(4 * np.pi / 5), tensor(6 * torch.pi / 5),
                  tensor(8 * torch.pi / 5)]
        rotation_matrix = get_2d_rotation_matrix(time_steps, modulation_period)
        pentagon_static = torch.asarray([
            [self._get_x_position(angles[0], nulling_baseline), self._get_x_position(angles[1], nulling_baseline),
             self._get_x_position(angles[2], nulling_baseline),
             self._get_x_position(angles[3], nulling_baseline), self._get_x_position(angles[4], nulling_baseline)],
            [self._get_y_position(angles[0], nulling_baseline), self._get_y_position(angles[1], nulling_baseline),
             self._get_y_position(angles[2], nulling_baseline),
             self._get_y_position(angles[3], nulling_baseline), self._get_y_position(angles[4], nulling_baseline)]],
            dtype=torch.float32)
        collector_positions = torch.einsum('ijl,jk->ikl', rotation_matrix,
                                           pentagon_static)
        return collector_positions.swapaxes(0, 2)

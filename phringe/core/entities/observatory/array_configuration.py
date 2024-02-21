from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import astropy.units
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from pydantic import BaseModel

from phringe.util.helpers import Coordinates
from phringe.util.matrix import get_2d_rotation_matrix


class ArrayConfigurationEnum(Enum):
    """Enum representing the different array configuration types.
    """
    EMMA_X_CIRCULAR_ROTATION = 'emma-x-circular-rotation'
    EMMA_X_DOUBLE_STRETCH = 'emma-x-double-stretch'
    EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION = 'equilateral-triangle-circular-rotation'
    REGULAR_PENTAGON_CIRCULAR_ROTATION = 'regular-pentagon-circular-rotation'


class ArrayConfiguration(ABC, BaseModel):
    """Class representation of a collector array configuration.

    :param baseline_length: The length of the baseline
    :param type: The type of the array configuration
    """
    baseline_length: Any = None
    type: Any = None
    collector_coordinates: Any = None

    @abstractmethod
    def get_collector_coordinates(
            self,
            time_steps: np.ndarray,
            modulation_period: Quantity,
            baseline_ratio: int
    ) -> np.ndarray:
        """Return time-dependent x- and y-coordinates of the collectors.

        :param time_steps: The time steps for which the collector positions are calculated
        :param modulation_period: The modulation period of the array
        :param baseline_ratio: The baseline ratio of the array
        :return: The array containing the coordinates of the collectors for each time step
        """
        pass


class EmmaXCircularRotation(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION

    def get_collector_coordinates(self,
                                  time_steps: np.ndarray,
                                  modulation_period: Quantity,
                                  baseline_ratio: int) -> Coordinates:
        rotation_matrix = get_2d_rotation_matrix(time_steps, modulation_period)
        emma_x_static = self.baseline_length / 2 * np.array(
            [[baseline_ratio, baseline_ratio, -baseline_ratio, -baseline_ratio], [1, -1, -1, 1]])
        collector_positions = np.einsum('ijl,jk->ikl', rotation_matrix, emma_x_static)
        collector_coordinates = np.zeros(len(time_steps), dtype=object)
        for i in range(len(time_steps)):
            collector_coordinates[i] = Coordinates(collector_positions[0, :, i], collector_positions[1, :, i])
        return collector_coordinates


class EmmaXDoubleStretch(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with double stretching of the array.
    """
    type: Any = ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH

    def get_collector_coordinates(self,
                                  time_steps: np.ndarray,
                                  modulation_period: Quantity,
                                  baseline_ratio: int) -> np.ndarray:
        emma_x_static = self.baseline_length / 2 * np.array(
            [[baseline_ratio, baseline_ratio, -baseline_ratio, -baseline_ratio], [1, -1, -1, 1]])
        # TODO: fix calculations
        collector_positions = emma_x_static * (1 + (2 * self.baseline_length) / self.baseline_length * np.sin(
            2 * np.pi * u.rad / modulation_period * time_steps))
        return Coordinates(collector_positions[0], collector_positions[1])


class EquilateralTriangleCircularRotation(ArrayConfiguration):
    """Class representation of an equilateral triangle configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION

    def get_collector_coordinates(self,
                                  time_steps: np.ndarray,
                                  modulation_period: Quantity,
                                  baseline_ratio: int) -> np.ndarray:
        height = np.sqrt(3) / 2 * self.baseline_length
        height_to_center = height / 3
        rotation_matrix = get_2d_rotation_matrix(time_steps, modulation_period)

        equilateral_triangle_static = np.array(
            [[0, self.baseline_length.value / 2, -self.baseline_length.value / 2],
             [height.value - height_to_center.value, -height_to_center.value, -height_to_center.value]])
        collector_positions = np.einsum('ijl,jk->ikl', rotation_matrix,
                                        equilateral_triangle_static) * self.baseline_length.unit
        collector_coordinates = np.zeros(len(time_steps), dtype=object)
        for i in range(len(time_steps)):
            collector_coordinates[i] = Coordinates(collector_positions[0, :, i], collector_positions[1, :, i])
        return collector_coordinates


class RegularPentagonCircularRotation(ArrayConfiguration):
    """Class representation of a regular pentagon configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION

    def _get_x_position(self, angle) -> astropy.units.Quantity:
        """Return the x position.

        :param angle: The angle at which the collector is located
        :return: The x position
        """
        return 0.851 * self.baseline_length.value * np.cos(angle)

    def _get_y_position(self, angle) -> astropy.units.Quantity:
        """Return the y position.

        :param angle: The angle at which the collector is located
        :return: The y position
        """
        return 0.851 * self.baseline_length.value * np.sin(angle)

    def get_collector_coordinates(self,
                                  time_steps: np.ndarray,
                                  modulation_period: Quantity,
                                  baseline_ratio: int) -> np.ndarray:
        angles = [0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5]
        rotation_matrix = get_2d_rotation_matrix(time_steps, modulation_period)
        pentagon_static = np.array([
            [self._get_x_position(angles[0]), self._get_x_position(angles[1]), self._get_x_position(angles[2]),
             self._get_x_position(angles[3]), self._get_x_position(angles[4])],
            [self._get_y_position(angles[0]), self._get_y_position(angles[1]), self._get_y_position(angles[2]),
             self._get_y_position(angles[3]), self._get_y_position(angles[4])]])
        collector_positions = np.einsum('ijl,jk->ikl', rotation_matrix,
                                        pentagon_static) * self.baseline_length.unit
        collector_coordinates = np.zeros(len(time_steps), dtype=object)
        for i in range(len(time_steps)):
            collector_coordinates[i] = Coordinates(collector_positions[0, :, i], collector_positions[1, :, i])
        return collector_coordinates

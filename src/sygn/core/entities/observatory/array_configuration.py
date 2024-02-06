from abc import ABC
from enum import Enum
from typing import Any

import astropy.units
import numpy as np
from pydantic import BaseModel


class ArrayConfigurationEnum(Enum):
    """Enum representing the different array configuration types.
    """
    EMMA_X_CIRCULAR_ROTATION = 'emma-x-circular-rotation'
    EMMA_X_DOUBLE_STRETCH = 'emma-x-double-stretch'
    EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION = 'equilateral-triangle-circular-rotation'
    REGULAR_PENTAGON_CIRCULAR_ROTATION = 'regular-pentagon-circular-rotation'


class ArrayConfiguration(ABC, BaseModel):
    """Class representation of a collector array configuration.
    """
    baseline_length: Any = None
    type: Any = None

    # @abstractmethod
    # def get_collector_positions(self, time: astropy.units.Quantity) -> Coordinates:
    #     """Return an array containing the time-dependent x- and y-coordinates of the collectors.
    #
    #     :param time: Time variable in seconds
    #     :return: An array containing the coordinates.
    #     """
    #     pass


class EmmaXCircularRotation(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION

    # def get_collector_positions(self, times: np.ndarray) -> Coordinates:
    #     rotation_matrix = get_2d_rotation_matrix(times, self.modulation_period)
    #     emma_x_static = self.baseline / 2 * np.array(
    #         [[self.baseline_ratio, self.baseline_ratio, -self.baseline_ratio, -self.baseline_ratio], [1, -1, -1, 1]])
    #     collector_positions = np.matmul(rotation_matrix, emma_x_static)
    #     return Coordinates(collector_positions[0], collector_positions[1])


class EmmaXDoubleStretch(ArrayConfiguration):
    """Class representation of the Emma-X array configuration with double stretching of the array.
    """
    type: Any = ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH

    # def get_collector_positions(self, time: float) -> Coordinates:
    #     emma_x_static = self.baseline / 2 * np.array(
    #         [[self.baseline_ratio, self.baseline_ratio, -self.baseline_ratio, -self.baseline_ratio], [1, -1, -1, 1]])
    #     # TODO: fix calculations
    #     collector_positions = emma_x_static * (1 + (2 * self.baseline) / self.baseline * np.sin(
    #         2 * np.pi * u.rad / self.modulation_period * time))
    #     return Coordinates(collector_positions[0], collector_positions[1])


class EquilateralTriangleCircularRotation(ArrayConfiguration):
    """Class representation of an equilateral triangle configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION

    # def get_collector_positions(self, time: float) -> Coordinates:
    #     height = np.sqrt(3) / 2 * self.baseline
    #     height_to_center = height / 3
    #     rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)
    #
    #     equilateral_triangle_static = np.array(
    #         [[0, self.baseline.value / 2, -self.baseline.value / 2],
    #          [height.value - height_to_center.value, -height_to_center.value, -height_to_center.value]])
    #     collector_positions = np.matmul(rotation_matrix, equilateral_triangle_static) * self.baseline.unit
    #     return Coordinates(collector_positions[0], collector_positions[1])


class RegularPentagonCircularRotation(ArrayConfiguration):
    """Class representation of a regular pentagon configuration with circular rotation of the array.
    """
    type: Any = ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION

    def _x(self, angle) -> astropy.units.Quantity:
        """Return the x position.

        :param angle: The angle at which the collector is located
        :return: The x position
        """
        return 0.851 * self.baseline_length.value * np.cos(angle)

    def _y(self, angle) -> astropy.units.Quantity:
        """Return the y position.

        :param angle: The angle at which the collector is located
        :return: The y position
        """
        return 0.851 * self.baseline_length.value * np.sin(angle)

    # def get_collector_positions(self, time: float) -> Coordinates:
    #     angles = [0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5]
    #     rotation_matrix = get_2d_rotation_matrix(time, self.modulation_period)
    #     pentagon_static = np.array([
    #         [self._x(angles[0]), self._x(angles[1]), self._x(angles[2]), self._x(angles[3]), self._x(angles[4])],
    #         [self._y(angles[0]), self._y(angles[1]), self._y(angles[2]), self._y(angles[3]), self._y(angles[4])]])
    #     collector_positions = np.matmul(rotation_matrix, pentagon_static) * self.baseline.unit
    #     return Coordinates(collector_positions[0], collector_positions[1])

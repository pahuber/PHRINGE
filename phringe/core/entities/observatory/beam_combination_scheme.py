from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np


class BeamCombinationSchemeEnum(Enum):
    """Enum representing the different beam combination schemes.
    """
    DOUBLE_BRACEWELL = 'double-bracewell'
    KERNEL_3 = 'kernel-3'
    KERNEL_4 = 'kernel-4'
    KERNEL_5 = 'kernel-5'


class BeamCombinationScheme(ABC):
    """Class representation of a beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = None

    def __init__(self):
        """Constructor method.
        """
        super().__init__()
        self.number_of_inputs = self.get_beam_combination_transfer_matrix().shape[1]
        self.number_of_outputs = self.get_beam_combination_transfer_matrix().shape[0]
        self.number_of_differential_outputs = len(self.get_differential_output_pairs())

    @abstractmethod
    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        """Return the beam combination transfer matrix.

        :return: An array representing the bea combination transfer matrix
        """
        pass

    @abstractmethod
    def get_differential_output_pairs(self) -> list:
        """Return the pairs of indices of the intensity response vector that make up one of the differential outputs.

        :return: List of tuples containing the pairs of indices
        """
        pass


class DoubleBracewell(BeamCombinationScheme):
    """Class representation of a double Bracewell beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / np.sqrt(4) * np.array([[0, 0, np.sqrt(2), np.sqrt(2)],
                                          [np.sqrt(2), np.sqrt(2), 0, 0],
                                          [1, -1, -np.exp(1j * np.pi / 2), np.exp(1j * np.pi / 2)],
                                          [1, -1, np.exp(1j * np.pi / 2), -np.exp(1j * np.pi / 2)]])

    def get_differential_output_pairs(self) -> list:
        return [(2, 3)]


class Kernel3(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinationSchemeEnum.KERNEL_3.value

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / np.sqrt(3) * np.array([[1, 1, 1],
                                          [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)],
                                          [1, np.exp(4j * np.pi / 3), np.exp(2j * np.pi / 3)]])

    def get_differential_output_pairs(self) -> list:
        return [(1, 2)]


class Kernel4(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinationSchemeEnum.KERNEL_4.value

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        exp_plus = np.exp(1j * np.pi / 2)
        exp_minus = np.exp(-1j * np.pi / 2)
        return 1 / 4 * np.array([[2, 2, 2, 2],
                                 [1 + exp_plus, 1 - exp_plus, -1 + exp_plus, -1 - exp_plus],
                                 [1 - exp_minus, -1 - exp_minus, 1 + exp_minus, -1 + exp_minus],
                                 [1 + exp_plus, 1 - exp_plus, -1 - exp_plus, -1 + exp_plus],
                                 [1 - exp_minus, -1 - exp_minus, -1 + exp_minus, 1 + exp_minus],
                                 [1 + exp_plus, -1 - exp_plus, 1 - exp_plus, -1 + exp_plus],
                                 [1 - exp_minus, -1 + exp_minus, -1 - exp_minus, 1 + exp_minus]])

    def get_differential_output_pairs(self) -> list:
        return [(1, 2), (3, 4), (5, 6)]


class Kernel5(BeamCombinationScheme):
    """Class representation of a Kernel nulling beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinationSchemeEnum.KERNEL_5.value

    def _get_exp_function(self, number: int) -> float:
        """Return the exponent.

        :param number: The number in the numerator
        :return: The exponent
        """
        return np.exp(1j * number * np.pi / 5)

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / np.sqrt(5) * np.array([[1, 1, 1, 1, 1],
                                          [1, self._get_exp_function(2), self._get_exp_function(4),
                                           self._get_exp_function(6), self._get_exp_function(8)],
                                          [1, self._get_exp_function(4), self._get_exp_function(8),
                                           self._get_exp_function(2), self._get_exp_function(6)],
                                          [1, self._get_exp_function(6), self._get_exp_function(2),
                                           self._get_exp_function(8), self._get_exp_function(4)],
                                          [1, self._get_exp_function(8), self._get_exp_function(6),
                                           self._get_exp_function(4), self._get_exp_function(2)]])

    def get_differential_output_pairs(self) -> list:
        return [(1, 4), (2, 3)]

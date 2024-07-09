from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import sqrt, tensor, exp, pi


class BeamCombinerEnum(Enum):
    """Enum representing the different beam combiners.
    """
    DOUBLE_BRACEWELL = 'double-bracewell'
    KERNEL_3 = 'kernel-3'
    KERNEL_4 = 'kernel-4'
    KERNEL_5 = 'kernel-5'


class BeamCombiner(ABC):
    """Class representation of a beam combiner.

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
        """Return the beam combination transfer matrix as an array of shape N_outputs x N_collectors.

        :return: An array representing the bea combination transfer matrix
        """
        pass

    @abstractmethod
    def get_differential_output_pairs(self) -> list:
        """Return the pairs of indices of the intensity response vector that make up one of the differential outputs.

        :return: List of tuples containing the pairs of indices
        """
        pass


class DoubleBracewell(BeamCombiner):
    """Class representation of a double Bracewell beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinerEnum.DOUBLE_BRACEWELL.value

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / sqrt(tensor(4)) * torch.tensor([[tensor(0), tensor(0), sqrt(tensor(2)), sqrt(tensor(2))],
                                                   [sqrt(tensor(2)), sqrt(tensor(2)), tensor(0), tensor(0)],
                                                   [tensor(1), tensor(-1), -exp(tensor(1j * pi / 2)),
                                                    exp(tensor(1j * pi / 2))],
                                                   [tensor(1), tensor(-1), exp(tensor(1j * pi / 2)),
                                                    -exp(tensor(1j * pi / 2))]], dtype=torch.complex64)

    def get_differential_output_pairs(self) -> list:
        return [(2, 3)]


class Kernel3(BeamCombiner):
    """Class representation of a Kernel nulling beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinerEnum.KERNEL_3.value

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / sqrt(tensor(3)) * tensor([[1, 1, 1],
                                             [1, exp(tensor(2j * pi / 3)), exp(tensor(4j * pi / 3))],
                                             [1, exp(tensor(4j * pi / 3)), exp(tensor(2j * pi / 3))]],
                                            dtype=torch.complex64)

    def get_differential_output_pairs(self) -> list:
        return [(1, 2)]


class Kernel4(BeamCombiner):
    """Class representation of a Kernel nulling beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinerEnum.KERNEL_4.value

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        exp_plus = exp(tensor(1j * pi / 2))
        exp_minus = exp(tensor(-1j * pi / 2))
        return 1 / 4 * torch.asarray([[2, 2, 2, 2],
                                      [1 + exp_plus, 1 - exp_plus, -1 + exp_plus, -1 - exp_plus],
                                      [1 - exp_minus, -1 - exp_minus, 1 + exp_minus, -1 + exp_minus],
                                      [1 + exp_plus, 1 - exp_plus, -1 - exp_plus, -1 + exp_plus],
                                      [1 - exp_minus, -1 - exp_minus, -1 + exp_minus, 1 + exp_minus],
                                      [1 + exp_plus, -1 - exp_plus, 1 - exp_plus, -1 + exp_plus],
                                      [1 - exp_minus, -1 + exp_minus, -1 - exp_minus, 1 + exp_minus]],
                                     dtype=torch.complex64)

    def get_differential_output_pairs(self) -> list:
        return [(1, 2), (3, 4), (5, 6)]


class Kernel5(BeamCombiner):
    """Class representation of a Kernel nulling beam combination scheme.

    :param type: The type of the beam combination scheme
    """
    type: Any = BeamCombinerEnum.KERNEL_5.value

    def _get_exp_function(self, number: int) -> float:
        """Return the exponent.

        :param number: The number in the numerator
        :return: The exponent
        """
        return exp(tensor(1j * number * pi / 5))

    def get_beam_combination_transfer_matrix(self) -> np.ndarray:
        return 1 / sqrt(tensor(5)) * torch.asarray([[1, 1, 1, 1, 1],
                                                    [1, self._get_exp_function(2), self._get_exp_function(4),
                                                     self._get_exp_function(6), self._get_exp_function(8)],
                                                    [1, self._get_exp_function(4), self._get_exp_function(8),
                                                     self._get_exp_function(2), self._get_exp_function(6)],
                                                    [1, self._get_exp_function(6), self._get_exp_function(2),
                                                     self._get_exp_function(8), self._get_exp_function(4)],
                                                    [1, self._get_exp_function(8), self._get_exp_function(6),
                                                     self._get_exp_function(4), self._get_exp_function(2)]],
                                                   dtype=torch.complex64)

    def get_differential_output_pairs(self) -> list:
        return [(1, 4), (2, 3)]

from abc import abstractmethod, ABC
from typing import Any, Union

import numpy as np
import torch
from astropy.units import Quantity
from numpy.random import normal
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy.fft import irfft, fftshift
from torch import Tensor

from phringe.core.base_entity import BaseEntity


class BasePerturbation(ABC, BaseEntity):
    rms: Union[str, float, Quantity] = None
    color: str = None
    _has_manually_set_time_series: bool = False

    @field_validator('color')
    def _validate_color(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the color input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The color
        """
        if value not in ['white', 'pink', 'brown']:
            raise ValueError(f'{value} is not a valid input for {info.field_name}. Must be one of white, pink, brown.')
        return value

    @property
    @abstractmethod
    def _time_series(self) -> Union[Tensor, None]:
        pass

    def _get_color_coeff(self) -> int:
        match self.color:
            case 'white':
                coeff = 0
            case 'pink':
                coeff = 1
            case 'brown':
                coeff = 2
            case _:
                coeff = None
        return coeff

    def _calculate_time_series_from_psd(
            self,
            coeff: int,
            modulation_period: float,
            number_of_simulation_time_steps: int
    ) -> np.ndarray:

        freq_cutoff_low = 1 / modulation_period
        freq_cutoff_high = 1e3
        freq = np.linspace(freq_cutoff_low, freq_cutoff_high, number_of_simulation_time_steps)
        omega = 2 * np.pi * freq

        ft = normal(loc=0, scale=(1 / omega) ** (coeff / 2)) + 1j * normal(loc=0, scale=(1 / omega) ** (coeff / 2))

        ft_total = np.concatenate((np.conjugate(np.flip(ft)), ft))
        time_series = irfft(fftshift(ft_total), n=number_of_simulation_time_steps)

        time_series /= np.sqrt(np.mean(time_series ** 2))

        if np.mean(time_series) > 0:
            time_series -= 1
        else:
            time_series += 1
        time_series /= np.sqrt(np.mean(time_series ** 2))
        time_series *= self.rms

        return torch.tensor(time_series, dtype=torch.float32, device=self._phringe._device)

    def set_time_series(self, time_series: Any):
        # TODO: implement set time series correctly
        self._time_series = time_series
        self._has_manually_set_time_series = True

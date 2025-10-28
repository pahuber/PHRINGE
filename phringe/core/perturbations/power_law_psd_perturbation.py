from typing import Union, Any

import astropy.units as u
import numpy as np
import torch
from astropy.units import Quantity
from numpy.random import normal
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy.fft import irfft, fftshift
from torch import Tensor

from phringe.core.perturbations.base_perturbation import BasePerturbation
from phringe.io.validation import validate_quantity_units


class PowerLawPSDPerturbation(BasePerturbation):
    coefficient: int
    rms: Union[str, float, Quantity]
    chromatic: bool = False
    _time_series: Union[Tensor, None] = None

    @field_validator('rms')
    def _validate_rms(cls, value: Any, info: ValidationInfo) -> float:
        return validate_quantity_units(
            value=value,
            field_name=info.field_name,
            unit_equivalency=(u.percent, u.radian, u.meter,)
        )

    @property
    def time_series(self) -> Tensor:
        return self._get_time_series()

    def _get_random_time_series_from_psd(self) -> Tensor:

        freq_cutoff_low = 1 / self._phringe._observation.modulation_period
        freq_cutoff_high = 1e3
        freq = np.linspace(freq_cutoff_low, freq_cutoff_high, len(self._phringe.simulation_time_steps))
        omega = 2 * np.pi * freq

        ft = (
                normal(loc=0, scale=(1 / omega) ** (self.coefficient / 2)) +
                1j * normal(loc=0, scale=(1 / omega) ** (self.coefficient / 2))
        )

        ft_total = np.concatenate((np.conjugate(np.flip(ft)), ft))

        time_series = irfft(fftshift(ft_total), n=len(self._phringe.simulation_time_steps))

        time_series /= np.sqrt(np.mean(time_series ** 2))

        if np.mean(time_series) > 0:
            time_series -= 1
        else:
            time_series += 1
        time_series /= np.sqrt(np.mean(time_series ** 2))
        time_series *= self.rms

        return torch.tensor(time_series, dtype=torch.float32, device=self._phringe._device)

    def _get_time_series(self) -> Tensor:

        num_inputs = self._phringe._instrument.number_of_inputs
        num_time_steps = len(self._phringe.simulation_time_steps)
        device = self._phringe._device

        if not self.chromatic:
            time_series = torch.zeros((num_inputs, num_time_steps), dtype=torch.float32, device=device)

            for k in range(num_inputs):
                time_series[k] = self._get_random_time_series_from_psd()

            return time_series

        # Add wavelength dimension and scale by 2pi/lambda if chromatic
        else:
            wavelengths = self._phringe._instrument.wavelength_bin_centers
            num_wl = len(wavelengths)
            bounds = self._phringe._instrument.wavelength_bands_boundaries
            num_bands = len(bounds) + 1

            time_series = torch.zeros((num_inputs, num_wl, num_time_steps,), dtype=torch.float32, device=device)

            for j in range(num_bands):

                # TODO: If multiple bands, is total RMS still correct like this?
                ts_per_band = torch.zeros((num_inputs, num_wl, num_time_steps,), dtype=torch.float32, device=device)

                for k in range(num_inputs):
                    ts_per_band[k] = self._get_random_time_series_from_psd()

                if num_bands == 1:
                    index_low = 0
                    index_up = len(wavelengths)
                elif j == 0:
                    index_low = 0
                    index_up = (torch.abs(wavelengths - bounds[j])).argmin()
                elif j == num_bands - 1:
                    index_low = (torch.abs(wavelengths - bounds[j - 1])).argmin()
                    index_up = len(wavelengths)
                else:
                    index_low = (torch.abs(wavelengths - bounds[j - 1])).argmin()
                    index_up = (torch.abs(wavelengths - bounds[j])).argmin()

                time_series[:, index_low:index_up, :] = (
                        2 * np.pi * ts_per_band[:, index_low:index_up, :] / wavelengths[None, index_low:index_up, None]
                )

            return time_series

from typing import Tuple, Any, Union

import numpy as np
import torch
from astropy import units as u
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from sympy import Matrix, lambdify
from sympy import symbols, Symbol, exp, I, pi, cos, sin, Abs, sqrt
from torch import Tensor

from phringe.core.base_entity import BaseEntity
from phringe.io.validation import validate_quantity_units


class Instrument(BaseEntity):
    """Class representing the instrument.

    Parameters
    ----------
    aperture_diameter : str or float or Quantity
        The aperture diameter in meters.
    array_configuration_matrix : Tensor
        The array configuration matrix.
    baseline_max : str or float or Quantity
        The max baseline in meters.
    baseline_min : str or float or Quantity
        The min baseline in meters.
    complex_amplitude_transfer_matrix : Tensor
        The complex amplitude transfer matrix.
    perturbations : Perturbations
        The perturbations.
    quantum_efficiency : float
        The quantum efficiency.
    sep_at_max_mod_eff : list
        The separation at max modulation efficiency.
    spectral_resolving_power : int
        The spectral resolving power.
    throughput : float
        The throughput.
    wavelength_min : str or float or Quantity
        The min wavelength in meters.
    wavelength_max : str or float or Quantity
        The max wavelength in meters.

    Attributes
    ----------
    number_of_inputs : int
        The number of inputs.
    number_of_outputs : int
        The number of outputs.
    response : Tensor
        The intensity response of the instrument.

    """
    aperture_diameter: Union[str, float, Quantity]
    array_configuration_matrix: Matrix
    nulling_baseline_max: Union[str, float, Quantity]
    nulling_baseline_min: Union[str, float, Quantity]
    complex_amplitude_transfer_matrix: Matrix
    kernels: Matrix
    quantum_efficiency: float
    spectral_resolving_power: int
    throughput: float
    wavelength_bands_boundaries: list
    wavelength_min: Union[str, float, Quantity]
    wavelength_max: Union[str, float, Quantity]
    amplitude_perturbation: Any = None
    phase_perturbation: Any = None
    polarization_perturbation: Any = None
    number_of_inputs: int = None
    number_of_outputs: int = None
    response: Tensor = None
    _torch_func_dict: dict = None

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self.number_of_inputs = self.complex_amplitude_transfer_matrix.shape[1]
        self.number_of_outputs = self.complex_amplitude_transfer_matrix.shape[0]

        def _torch_sqrt(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.sqrt(x)

        def _torch_exp(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.exp(x)

        self._torch_func_dict = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': _torch_exp,
            'log': torch.log,
            'sqrt': _torch_sqrt,
            'transpose': torch.transpose,
        }
        self._calc_symbolic_response()
        self._calc_lambdified_response()

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "_phringe":
            if self.amplitude_perturbation is not None:
                self.amplitude_perturbation._phringe = self._phringe
            if self.phase_perturbation is not None:
                self.phase_perturbation._phringe = self._phringe
            if self.polarization_perturbation is not None:
                self.polarization_perturbation._phringe = self._phringe

    @field_validator('aperture_diameter')
    def _validate_aperture_diameter(cls, value: Any, info: ValidationInfo) -> Tensor:
        """Validate the aperture diameter input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The aperture diameter in units of length
        """
        return torch.tensor(
            validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)),
            dtype=torch.float32
        )

    @field_validator('nulling_baseline_min')
    def _validate_nulling_baseline_min(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the nulling baseline min input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The min nulling baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('nulling_baseline_max')
    def _validate_nulling_baseline_max(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the nulling baseline max input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The max nulling baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('wavelength_bands_boundaries')
    def _validate_wavelength_bands_boundaries(cls, value: Any, info: ValidationInfo) -> list:
        """Validate the wavelength bands boundaries input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The wavelength bands boundaries in units of length
        """
        return [
            validate_quantity_units(
                value=boundary,
                field_name=info.field_name,
                unit_equivalency=(u.m,)
            )
            for boundary in value
        ]

    @field_validator('wavelength_min')
    def _validate_wavelength_min(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the wavelength range lower limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The lower wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('wavelength_max')
    def _validate_wavelength_max(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the wavelength range upper limit input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The upper wavelength range limit in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @property
    def _field_of_view(self):
        return self._get_field_of_view()

    @property
    def _number_of_simulation_time_steps(self):
        return len(self._simulation_time_steps)

    @property
    def _wavelength_bins(self) -> Tuple[Tensor, Tensor]:
        """Return the wavelength bin centers and widths.

        :return: A tuple containing the wavelength bin centers and widths
        """
        return self._get_wavelength_bins()

    @property
    def wavelength_bin_centers(self) -> Tensor:
        """Return the wavelength bin centers.

        :return: An array containing the wavelength bin centers
        """
        return self._wavelength_bins[0]

    @property
    def wavelength_bin_widths(self) -> Tensor:
        """Return the wavelength bin widths.

        :return: An array containing the wavelength bin widths
        """
        return self._wavelength_bins[1]

    @property
    def wavelength_bin_edges(self) -> Tensor:
        """Return the wavelength bin edges.

        :return: An array containing the wavelength bin edges
        """
        return torch.concatenate(
            (
                self.wavelength_bin_centers - self.wavelength_bin_widths / 2,
                self.wavelength_bin_centers[-1:] + self.wavelength_bin_widths[-1:] / 2
            )
        )

    def _calc_lambdified_response(self):
        """Calculate the lambdified instrument response.
        """
        # Build substitution dictionary for the symbolic expressions for fixed quantities
        subs = {}

        for j in range(self.number_of_outputs):
            for k in range(self.number_of_inputs):
                subs[self._sym_catm[j, k]] = self.complex_amplitude_transfer_matrix[j, k]

        for i in range(2):
            for k in range(self.number_of_inputs):
                subs[self._sym_acm[i, k]] = self.array_configuration_matrix[i, k]

        subs[self._sym_ap_diam] = self.aperture_diameter

        for k in range(self.number_of_inputs):
            subs[self._sym_ampl[k]] = self._get_amplitude()

        # Calculate kernels
        response_vec = Matrix([self._response_symbolic[j] for j in range(self.number_of_outputs)])
        response_vec = response_vec.xreplace(subs)
        response_symbolic_kernels = self.kernels @ response_vec

        # Lambdify response kernels for torch
        self._response_kernels_torch = [
            lambdify(
                self._response_arg_symbols, response_symbolic_kernels[i, 0], [self._torch_func_dict]
            ) for i in range(response_symbolic_kernels.rows)
        ]

        # Lambdify response kernels for numpy
        self._response_kernels_numpy = [
            lambdify(
                self._response_arg_symbols, response_symbolic_kernels[i, 0], 'numpy'
            ) for i in range(response_symbolic_kernels.rows)
        ]

        # Lambdify full response for torch
        self._response_torch = [
            lambdify(
                self._response_arg_symbols,
                self._response_symbolic[j].xreplace(subs),
                [self._torch_func_dict]
            )
            for j in range(self.number_of_outputs)
        ]

        # Lambdify full response for numpy
        self._response_numpy = [
            lambdify(
                self._response_arg_symbols,
                self._response_symbolic[j].xreplace(subs),
                'numpy'
            )
            for j in range(self.number_of_outputs)
        ]

    def _calc_symbolic_response(self):
        """Return the symbolic intensity response using SymPy.
        """
        # Define symbols for symbolic expressions
        self._sym_catm = Matrix(
            self.number_of_outputs,
            self.number_of_inputs,
            lambda j, k: Symbol(f"catm_{j}_{k}", complex=True)
        )

        self._sym_acm = Matrix(
            2,
            self.number_of_inputs,
            lambda i, k: Symbol(f"acm_{i}_{k}", real=True)
        )

        self._sym_ap_diam = Symbol("ap_diam", real=True)
        self._sym_ampl = {k: Symbol(f'ampl_{k}', real=True) for k in range(self.number_of_inputs)}
        self._sym_ampl_pert = {k: Symbol(f'ampl_pert_{k}', real=True) for k in range(self.number_of_inputs)}
        self._sym_phase_pert = {k: Symbol(f'phase_pert_{k}', real=True) for k in range(self.number_of_inputs)}
        self._sym_pol_pert = {k: Symbol(f'pol_pert_{k}', real=True) for k in range(self.number_of_inputs)}
        (self._sym_time,
         self._sym_nulling_baseline,
         self._sym_wavelength,
         self._sym_modulation_time,
         self._sym_alpha_coord,
         self._sym_beta_coord) = (symbols('t b l tm alpha beta'))

        self._response_arg_symbols = [
            self._sym_time,
            self._sym_wavelength,
            self._sym_alpha_coord,
            self._sym_beta_coord,
            self._sym_modulation_time,
            self._sym_nulling_baseline,
            *self._sym_ampl_pert.values(),
            *self._sym_phase_pert.values(),
            *self._sym_pol_pert.values(),
        ]

        # Calculate complex amplitudes
        complex_ampl_x = {}
        complex_ampl_y = {}
        for k in range(self.number_of_inputs):
            phase = 2 * pi / self._sym_wavelength * (
                    self._sym_acm[0, k] * self._sym_alpha_coord + self._sym_acm[1, k] * self._sym_beta_coord) + \
                    self._sym_phase_pert[k]
            common = self._sym_ampl[k] * sqrt(pi) * (self._sym_ampl_pert[k] + 1) * exp(I * phase)
            complex_ampl_x[k] = common * cos(
                self._sym_pol_pert[k])  # cos(th[k] + pol_pert[k]) assuming th = 0 for all k
            complex_ampl_y[k] = common * sin(
                self._sym_pol_pert[k])  # sin(th[k] + pol_pert[k]) assuming th = 0 for all k

        # Calculate fov taper function to account for the fov-limiting effect of the single-mode fiber
        fov_taper = exp(-(pi ** 2 * (
                self._sym_alpha_coord ** 2 + self._sym_beta_coord ** 2) * self._sym_ap_diam ** 2 / self._sym_wavelength ** 2))

        # Calculate intensity response
        response_total = {}
        response_x = {}
        response_y = {}

        for j in range(self.number_of_outputs):
            response_x[j] = 0
            response_y[j] = 0
            for k in range(self.number_of_inputs):
                response_x[j] += self._sym_catm[j, k] * complex_ampl_x[k]
                response_y[j] += self._sym_catm[j, k] * complex_ampl_y[k]
            response_total[j] = (Abs(response_x[j]) ** 2 + Abs(response_y[j]) ** 2) * fov_taper

        self._response_symbolic = response_total

    def _get_amplitude(self) -> Tensor:
        """Return the amplitude of the instrument based on collecting area and throughput terms.

        Returns
        -------
        torch.Tensor
            The amplitude of the instrument.
        """
        return self.aperture_diameter / 2 * np.sqrt(self.throughput * self.quantum_efficiency)

    def _get_field_of_view(self) -> Tensor:
        """Return the field of view.

        Returns
        -------
        torch.Tensor
            Fields of view for all wavelength bins.

        """
        return self.wavelength_bin_centers / self.aperture_diameter

    def _get_wavelength_bins(self) -> Tuple[Tensor, Tensor]:
        """Return the wavelength bin centers and widths. The wavelength bin widths are calculated starting from the
        lower wavelength range limit. The bins are iteratively calculated and increase in size towards longer wavelength.
        If the uppermost bin does not fit into the remaining wavelength range, it is omitted.

        :return: A tuple containing the wavelength bin centers and widths
        """
        current_min_wavelength = self.wavelength_min
        wavelength_bin_centers = []
        wavelength_bin_widths = []

        while current_min_wavelength <= self.wavelength_max:
            center_wavelength = current_min_wavelength / (1 - 1 / (2 * self.spectral_resolving_power))
            bin_width = 2 * (center_wavelength - current_min_wavelength)
            if (center_wavelength + bin_width / 2 <= self.wavelength_max):
                wavelength_bin_centers.append(center_wavelength)
                wavelength_bin_widths.append(bin_width)
                current_min_wavelength = center_wavelength + bin_width / 2

            # If there is not enough space for the last bin, leave it away
            else:
                wavelength_bin_centers = wavelength_bin_centers[:-1]
                wavelength_bin_widths = wavelength_bin_widths[:-1]
                break

        return (
            torch.asarray(wavelength_bin_centers, dtype=torch.float32, device=self._phringe._device),
            torch.asarray(wavelength_bin_widths, dtype=torch.float32, device=self._phringe._device)
        )

    def get_response(
            self,
            times: Tensor,
            wavelength_bin_centers: Tensor,
            x_sky_coordinates: Tensor,
            y_sky_coordinates: Tensor,
            modulation_period: float,
            nulling_baseline: float,
            kernels: bool,
            amplitude_perturbation: Tensor = None,
            phase_perturbation: Tensor = None,
            polarization_perturbation: Tensor = None,
    ):
        """Return the intensity response of the instrument as a tensor.

        Parameters
        ----------
        times : Tensor
            Times at which the response should be evaluated.
        wavelength_bin_centers : Tensor
            Wavelength bin centers.
        x_sky_coordinates : Tensor
            Sky coordinates along x (alpha).
        y_sky_coordinates : Tensor
            Sky coordinates along y (beta).
        modulation_period : float
            Modulation period.
        nulling_baseline : float
            Nulling baseline.
        kernels : bool
            Whether to return the kernels or the full response.
        amplitude_perturbation : Tensor, optional
            Amplitude perturbation for each input.
            Expected shape: (..., number_of_inputs)
        phase_perturbation : Tensor, optional
            Phase perturbation for each input.
            Expected shape: (..., number_of_inputs)
        polarization_perturbation : Tensor, optional
            Polarization perturbation for each input.
            Expected shape: (..., number_of_inputs)

        Returns
        -------
        Tensor
            Instrument response tensor.
        """
        device = self._phringe._device
        n_in = self.number_of_inputs
        n_t = times.shape[0]
        n_w = wavelength_bin_centers.shape[0]

        # Default perturbations
        if amplitude_perturbation is None:
            amplitude_perturbation = torch.zeros((n_in, n_t), dtype=torch.float32, device=device)[
                :, None, :, None, None]

        if phase_perturbation is None:
            phase_perturbation = torch.zeros((n_in, n_w, n_t), dtype=torch.float32, device=device)[:, :, :, None, None]

        if polarization_perturbation is None:
            polarization_perturbation = torch.zeros((n_in, n_t), dtype=torch.float32, device=device)[
                :, None, :, None, None]

        args = [
            times,
            wavelength_bin_centers,
            x_sky_coordinates,
            y_sky_coordinates,
            modulation_period,
            nulling_baseline,
            *[amplitude_perturbation[i] for i in range(n_in)],
            *[phase_perturbation[i] for i in range(n_in)],
            *[polarization_perturbation[i] for i in range(n_in)],
        ]

        # Evaluate all outputs
        if not kernels:
            response = torch.stack(
                [func(*args) for func in self._response_torch],
                dim=0
            )
        else:
            response = torch.stack(
                [func(*args) for func in self._response_kernels_torch],
                dim=0
            )

        return response

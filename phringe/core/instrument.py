from typing import Tuple, Any, Union

import numpy as np
import torch
from astropy import units as u
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from sympy import Matrix
from sympy import symbols, Symbol, exp, I, pi, cos, sin, Abs, lambdify, sqrt
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
        The response.

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

    def __init__(self, **data):
        super().__init__(**data)
        self.number_of_inputs = self.complex_amplitude_transfer_matrix.shape[1]
        self.number_of_outputs = self.complex_amplitude_transfer_matrix.shape[0]
        self.response = self._get_lambdafied_response()

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
    def _wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the wavelength bin centers and widths.

        :return: A tuple containing the wavelength bin centers and widths
        """
        return self._get_wavelength_bins()

    @property
    def wavelength_bin_centers(self) -> np.ndarray:
        """Return the wavelength bin centers.

        :return: An array containing the wavelength bin centers
        """
        return self._wavelength_bins[0]

    @property
    def wavelength_bin_widths(self) -> np.ndarray:
        """Return the wavelength bin widths.

        :return: An array containing the wavelength bin widths
        """
        return self._wavelength_bins[1]

    @property
    def wavelength_bin_edges(self) -> np.ndarray:
        """Return the wavelength bin edges.

        :return: An array containing the wavelength bin edges
        """
        return torch.concatenate(
            (
                self.wavelength_bin_centers - self.wavelength_bin_widths / 2,
                self.wavelength_bin_centers[-1:] + self.wavelength_bin_widths[-1:] / 2
            )
        )

    def _get_amplitude(self, device: torch.device) -> Tensor:
        return self.aperture_diameter / 2 * torch.sqrt(
            torch.tensor(self.throughput * self.quantum_efficiency, device=device)
        )

    def _get_field_of_view(self) -> Tensor:
        return self.wavelength_bin_centers / self.aperture_diameter

    def _get_lambdafied_response(self):
        # Define symbols for symbolic expressions
        catm = self.complex_amplitude_transfer_matrix
        acm = self.array_configuration_matrix
        ex = {}
        ey = {}
        a = {}
        da = {}
        dphi = {}
        th = {}
        dth = {}
        t, tm, b, l, alpha, beta = symbols('t tm b l alpha beta')

        # Define complex amplitudes
        for k in range(self.number_of_inputs):
            a[k] = Symbol(f'a_{k}', real=True)
            da[k] = Symbol(f'da_{k}', real=True)
            dphi[k] = Symbol(f'dphi_{k}', real=True)
            th[k] = Symbol(f'th_{k}', real=True)
            dth[k] = Symbol(f'dth_{k}', real=True)
            ex[k] = a[k] * sqrt(pi) * (da[k] + 1) * exp(
                I * (2 * pi / l * (acm[0, k] * alpha + acm[1, k] * beta) + dphi[k])) * cos(
                th[k] + dth[k])
            ey[k] = a[k] * sqrt(pi) * (da[k] + 1) * exp(
                I * (2 * pi / l * (acm[0, k] * alpha + acm[1, k] * beta) + dphi[k])) * sin(
                th[k] + dth[k])

        # Define intensity response and save the symbolic expression
        r = {}
        rx = {}
        ry = {}
        r_torch = {}
        r_numpy = {}

        self._symbolic_intensity_response = {}
        for j in range(self.number_of_outputs):
            rx[j] = 0
            ry[j] = 0
            for k in range(self.number_of_inputs):
                rx[j] += catm[j, k] * ex[k]
                ry[j] += catm[j, k] * ey[k]
            r[j] = Abs(rx[j]) ** 2 + Abs(ry[j]) ** 2
            self._symbolic_intensity_response[j] = r[j]

        # Compile the intensity response functions for numerical calculations and save the lambdified functions
        def _torch_sqrt(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.sqrt(x)

        def _torch_exp(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.exp(x)

        torch_func_dict = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': _torch_exp,
            'log': torch.log,
            'sqrt': _torch_sqrt
        }

        self._diff_ir_torch = {}
        self._diff_ir_numpy = {}
        self._ir_numpy = {}

        # Lambdify differential output for torch
        r_vec = Matrix([r[j] for j in range(self.number_of_outputs)])  # shape (n_out, 1)
        expr = self.kernels @ r_vec

        self._diff_ir_torch = [lambdify(
            [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
            expr[i, 0],
            [torch_func_dict]
        ) for i in range(expr.rows)]

        # Lambdify differential output for numpy
        self._diff_ir_numpy = [lambdify(
            [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
            expr[i, 0],
            'numpy'
        ) for i in range(expr.rows)]

        for j in range(self.number_of_outputs):
            # Lambdify intensity response for numpy used in model count generation with photon noise
            self._ir_numpy[j] = lambdify(
                [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
                r[j],
                'numpy'
            )

            # Lambdify intensity response for torch used in major calculations
            r[j] = lambdify(
                [t, l, alpha, beta, tm, b, *a.values(), *da.values(), *dphi.values(), *th.values(), *dth.values(), ],
                r[j],
                [torch_func_dict]
            )

        return r

    def _get_wavelength_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the wavelength bin centers and widths. The wavelength bin widths are calculated starting from the
        wavelength lower range. As a consequence, the uppermost wavelength bin could be smaller than anticipated, in
        which case it is added to the second last bin width, so the last bin might be a bit larger than anticipated.

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

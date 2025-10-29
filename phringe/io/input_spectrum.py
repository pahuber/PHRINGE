from __future__ import annotations
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import astropy.units as u
import numpy as np
import spectres
import torch
from astropy.units import Quantity, Unit, UnitBase
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo
from torch import Tensor

from phringe.io.validation import validate_quantity_units


class InputSpectrum(BaseModel):
    """Class representation of an input spectral energy distribution (SED). Can be used to give a custom spectrum to a
    planet rather than the auto-generated blackbody spectrum.

    Parameters
    ----------
    sed_units : Any
        Units of the spectral energy distribution.
    wavelength_units : Any
        Units of the wavelengths.
    path_to_file : Path, optional
        Path to a text file containing the SED data. The file should have two columns:
        the first column for wavelengths and the second column for SED values. Comments marked with '#' are ignored.
    sed : np.ndarray, optional
        Array containing the SED values. Can be used to pass the SED values as an array directly, rather than importing
        them through a file.
    wavelengths : np.ndarray, optional
        Array containing the wavelengths corresponding to the SED values. Can be used to pass the wavelengths as an
        array directly, rather than importing them through a file.
    observed_planet_radius : Any, optional
        If the spectrum is given for a specific target, i.e. if sed_units do not contain "per steradian", this is the
        radius of that target planet. It is used to normalize the SED to solid angle.
    observed_host_star_distance : Any, optional
        If the spectrum is given for a specific target, i.e. if sed_units do not contain "per steradian", this is the
        distance between the host star and the instrument. It is used to normalize the SED to solid angle.
    """
    sed_units: Union[str, UnitBase]
    wavelength_units: Union[str, Unit]
    path_to_file: Path = None
    sed: np.ndarray = None
    wavelengths: np.ndarray = None
    observed_planet_radius: Union[str, float, Quantity] = None
    observed_host_star_distance: Union[str, float, Quantity] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        """Initialize the InputSpectrum object."""
        super().__init__(**data)

        # Check that either path_to_file or both sed and wavelengths are provided
        if self.path_to_file is None:
            if self.sed is None or self.wavelengths is None:
                raise ValueError("Either path_to_file or both sed and wavelengths must be provided.")
        else:
            if self.sed is not None or self.wavelengths is not None:
                raise ValueError("If path_to_file is provided, sed and wavelengths must be None.")

            # If only path_to_file is provided, read the file
            self._read_txt_file()
        # Check that if units contain /sr, a planet radius and star distance are given and vice-versa
        if u.sr not in self.sed_units.bases:
            if self.observed_planet_radius is None or self.observed_host_star_distance is None:
                raise ValueError(
                    "If SED is not given per steradian, the observed_planet_radius and observed_host_star_distance are required.")
        else:
            if self.observed_planet_radius is not None or self.observed_host_star_distance is not None:
                raise ValueError(
                    "If SED is given per steradian, the observed_planet_radius and observed_host_star_distance must be None.")

        self._convert_wavelengths_to_standard_units()
        self._convert_sed_to_standard_units()

    @field_validator('sed_units')
    def _validate_sed_units(cls, value: Any) -> u.UnitBase:
        """Validate the spectral energy distribution units input.

        Parameters
        ----------
        value : Any
            Value given as input

        Returns
        -------
        u.UnitBase
            The SED units
        """
        if isinstance(value, str):
            value = u.Unit(value)

        if not cls._is_sed_unit(value):
            raise ValueError(f"The provided SED units '{value}' are not valid SED units.")

        return value

    @field_validator('wavelength_units')
    def _validate_wavelength_units(cls, value: Any) -> u.UnitBase:
        """Validate the wavelength units input.

        Parameters
        ----------
        value : Any
            Value given as input

        Returns
        -------
        u.UnitBase
            The wavelength units
        """
        if isinstance(value, str):
            return u.Unit(value)

        if not value.is_equivalent(u.m) and not value.is_equivalent(u.Hz):
            raise ValueError(f"The provided wavelength units '{value}' are not valid wavelength units.")

        return value

    @field_validator('observed_planet_radius')
    def _validate_observed_planet_radius(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the observed planet radius input.

        Parameters
        ----------
        value : Any
            Value given as input
        info : ValidationInfo
            ValidationInfo object

        Returns
        -------
        float
            The observed planet radius in units of meters
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('observed_host_star_distance')
    def _validate_observed_host_star_distance(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the observed host star distance input.

        Parameters
        ----------
        value : Any
            Value given as input
        info : ValidationInfo
            ValidationInfo object

        Returns
        -------
        float
            The observed host star distance in units of meters
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @staticmethod
    def _is_sed_unit(unit: u.UnitBase) -> bool:
        """Check if the provided unit is a valid spectral energy distribution (SED) unit.

        Parameters
        ----------
        unit : u.UnitBase
            The unit to check.

        Returns
        -------
        bool
            True if the unit is a valid SED unit, False otherwise.
        """
        # Equivalencies:
        eq = []
        eq += u.dimensionless_angles()  # treat sr as dimensionless
        eq += u.spectral()  # λ ↔ ν ↔ E
        # spectral density (value here is arbitrary; only the unit matters)
        eq += u.spectral_density(1 * u.Hz)
        eq += u.spectral_density(1 * u.m)
        eq += u.spectral_density(1 * u.eV)

        # Canonical “SED-like” target families: energy or photons, per λ or per ν
        targets = [
            u.W / u.m ** 2 / u.Hz,  # F_ν
            u.W / u.m ** 2 / u.m,  # F_λ
            u.W / u.m ** 3,  # energy density per λ (no area)
            u.erg / u.s / u.cm ** 2 / u.Hz,  # CGS
            u.photon / u.s / u.m ** 2 / u.Hz,
            u.photon / u.s / u.m ** 2 / u.m,
        ]

        # Accept with or without steradian and with extra /m^2 if user means radiance vs flux
        variants = []
        for t in targets:
            variants += [t, t / u.sr, t * u.sr]

        return any(unit.is_equivalent(v, equivalencies=eq) for v in variants)

    def _convert_wavelengths_to_standard_units(self):
        """Convert wavelengths values to SI units (meters)."""
        if self.wavelength_units.is_equivalent(u.m):
            self.wavelengths = (self.wavelengths * self.wavelength_units).to(u.m).value
        else:
            self.wavelengths = (self.wavelengths * self.wavelength_units).to(u.m, equivalencies=u.spectral()).value

    def _convert_sed_to_standard_units(self):
        """Convert SED values to standard units (photons / s / m^3 / sr)."""
        if u.sr not in self.sed_units.bases:
            solid_angle = np.pi * (self.observed_planet_radius / self.observed_host_star_distance) ** 2
            self.sed /= solid_angle
            self.sed_units /= u.sr

        self.sed = (self.sed * self.sed_units).to(u.ph / u.s / u.m ** 3 / u.sr, equivalencies=u.spectral_density(
            self.wavelengths * u.m)).value

    def _read_txt_file(self):
        """Read the SED data from a text file."""
        content = np.loadtxt(self.path_to_file, usecols=(0, 1))
        self.sed = content[:, 1]
        self.wavelengths = content[:, 0]

    def get_spectral_energy_distribution(
            self,
            wavelength_bin_centers: Tensor,
            solid_angle: Tensor,
            device: torch.device
    ) -> Tensor:
        """Return the spectral energy distribution (SED) in units of ph / s / m^3, rebinned to the provided wavelength
        bins and scaled to the provided solid angle.

        Parameters
        ----------
        wavelength_bin_centers : Tensor
            Wavelength bin centers to rebin the SED to.
        solid_angle : Tensor
            Solid angle to scale the SED to.
        device : torch.device
            Device to perform the calculations on.

        Returns
        -------
        Tensor
            The binned and scaled spectral energy distribution in units of ph / s / m^3.
        """
        binned_spectral_flux_density = spectres.spectres(
            wavelength_bin_centers.cpu().numpy(),
            self.wavelengths,
            self.sed,
            fill=0,
            verbose=False
        ) * solid_angle
        return torch.asarray(binned_spectral_flux_density, dtype=torch.float32, device=device)

from typing import Any

import astropy
from astropy import units as u
from astropy.units import Quantity
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.core.entities.base_component import BaseComponent
from phringe.io.validators import validate_quantity_units


class Observation(BaseComponent, BaseModel):
    """Class representing the observation.

    :param total_integration_time: The total integration time
    :param exposure_time: The exposure time
    :param modulation_period: The modulation period
    :param baseline_ratio: The baseline ratio
    :param baseline_maximum: The maximum baseline
    :param baseline_minimum: The minimum baseline
    :param optimized_differential_output: The optimized differential output
    :param optimized_star_separation: The optimized star separation
    :param optimized_wavelength: The optimized wavelength
    """
    total_integration_time: str
    exposure_time: str
    modulation_period: str
    baseline_ratio: int
    baseline_maximum: str
    baseline_minimum: str
    optimized_differential_output: int
    optimized_star_separation: str
    optimized_wavelength: str

    @field_validator('baseline_minimum')
    def _validate_baseline_minimum(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the baseline minimum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The minimum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('baseline_maximum')
    def _validate_baseline_maximum(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the baseline maximum input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The maximum baseline in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('exposure_time')
    def _validate_exposure_time(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the exposure time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The exposure time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    @field_validator('modulation_period')
    def _validate_modulation_period(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the modulation period input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The modulation period in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    @field_validator('optimized_star_separation')
    def _validate_optimized_star_separation(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized star separation input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized star separation in its original units or as a string
        """
        if value == 'habitable-zone':
            return value
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m, u.arcsec))

    @field_validator('optimized_wavelength')
    def _validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized wavelength in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

        """Prepare the observation for the simulation."""
        pass

    @field_validator('total_integration_time')
    def _validate_total_integration_time(cls, value: Any, info: ValidationInfo) -> astropy.units.Quantity:
        """Validate the total integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The total integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    def prepare(self):
        """Prepare the observation for the simulation."""
        pass

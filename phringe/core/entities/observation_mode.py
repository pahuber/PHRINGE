from typing import Any

from astropy import units as u
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.io.validators import validate_quantity_units


class ObservationMode(BaseModel):
    """Class representing the observation mode.

    :param solar_ecliptic_latitude: The solar ecliptic latitude
    :param total_integration_time: The total integration time
    :param detector_integration_time: The detector integration time
    :param modulation_period: The modulation period
    :param baseline_ratio: The baseline ratio
    :param baseline_maximum: The maximum baseline
    :param baseline_minimum: The minimum baseline
    :param optimized_differential_output: The optimized differential output
    :param optimized_star_separation: The optimized star separation
    :param optimized_wavelength: The optimized wavelength
    """
    solar_ecliptic_latitude: Any
    total_integration_time: str
    detector_integration_time: str
    modulation_period: str
    optimized_differential_output: int
    optimized_star_separation: str
    optimized_wavelength: str

    @field_validator('detector_integration_time')
    def _validate_detector_integration_time(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the detector integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The detector integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,)).si.value

    @field_validator('modulation_period')
    def _validate_modulation_period(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the modulation period input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The modulation period in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,)).si.value

    @field_validator('optimized_star_separation')
    def _validate_optimized_star_separation(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the optimized star separation input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized star separation in its original units or as a string
        """
        if value == 'habitable-zone':
            return value
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,)).si.value

    @field_validator('optimized_wavelength')
    def _validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized wavelength in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)).si.value

    @field_validator('solar_ecliptic_latitude')
    def _validate_solar_ecliptic_latitude(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the solar ecliptic latitude input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The solar ecliptic latitude in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,)).si.value

    @field_validator('total_integration_time')
    def _validate_total_integration_time(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the total integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The total integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,)).si.value

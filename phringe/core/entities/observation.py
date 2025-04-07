from typing import Any, Union

from astropy import units as u
from astropy.units import Quantity
from phringe.core.base_entity import BaseEntity
from phringe.io.validators import validate_quantity_units
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo


class Observation(BaseEntity):
    """Class representing the observation mode.

    Parameters
    ----------
    detector_integration_time : str or float or Quantity
        The detector integration time in seconds.
    modulation_period : str or float or Quantity
        The modulation/rotation period of the array in seconds.
    optimized_differential_output : int
        Optimized differential output index. If the baseline is not set manually, it is set such that the transmission is optimum for the optimized differential output.
    optimized_star_separation : str or float or Quantity
        Optimized star separation in radians or the string 'habitable-zone'. If the baseline is not set manually, it is set such that the transmission is optimum for the optimized star separation.
    optimized_wavelength : str or float or Quantity
        Optimized wavelength in meters. If the baseline is not set manually, it is set such that the transmission is optimum for the optimized wavelength.
    solar_ecliptic_latitude : str or float or Quantity
        The solar ecliptic latitude in degrees. Used for the local zodi contribution calculation.
    total_integration_time : str or float or Quantity
        The total integration time in seconds.
    """
    detector_integration_time: Union[str, float, Quantity]
    modulation_period: Union[str, float, Quantity]
    optimized_differential_output: int
    optimized_star_separation: Union[str, float, Quantity]
    optimized_wavelength: Union[str, float, Quantity]
    solar_ecliptic_latitude: Union[str, float, Quantity]
    total_integration_time: Union[str, float, Quantity]

    @field_validator('detector_integration_time')
    def _validate_detector_integration_time(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the detector integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The detector integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    @field_validator('modulation_period')
    def _validate_modulation_period(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the modulation period input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The modulation period in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

    @field_validator('optimized_star_separation')
    def _validate_optimized_star_separation(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the optimized star separation input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized star separation in its original units or as a string
        """
        if value == 'habitable-zone':
            return value
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,))

    @field_validator('optimized_wavelength')
    def _validate_optimized_wavelength(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the optimized wavelength input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The optimized wavelength in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    @field_validator('solar_ecliptic_latitude')
    def _validate_solar_ecliptic_latitude(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the solar ecliptic latitude input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The solar ecliptic latitude in units of degrees
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    @field_validator('total_integration_time')
    def _validate_total_integration_time(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the total integration time input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The total integration time in units of time
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.s,))

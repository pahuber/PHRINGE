from typing import Any, Union

from astropy import units as u
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.core.base_entity import BaseEntity
from phringe.io.validation import validate_quantity_units
from phringe.util.baseline import OptimalNullingBaseline


class Observation(BaseEntity):
    """Class representing the observation mode.

    Parameters
    ----------
    detector_integration_time : str or float or Quantity
        The detector integration time in seconds.
    modulation_period : str or float or Quantity
        The modulation/rotation period of the array in seconds.
    nulling_baseline : str or float or Quantity or OptimalNullingBaseline
        The nulling baseline in meters or an optimized nulling baseline
    solar_ecliptic_latitude : str or float or Quantity
        The solar ecliptic latitude in degrees. Used for the local zodi contribution calculation.
    total_integration_time : str or float or Quantity
        The total integration time in seconds.
    """
    detector_integration_time: Union[str, float, Quantity]
    modulation_period: Union[str, float, Quantity]
    nulling_baseline: Union[str, float, Quantity, OptimalNullingBaseline]
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

    @field_validator('nulling_baseline')
    def _validate_nulling_baseline(cls, value: Any, info: ValidationInfo) -> Union[float, OptimalNullingBaseline]:
        """Validate the nulling baseline input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The nulling baseline in units of length
        """
        if isinstance(value, OptimalNullingBaseline):
            return value
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

    @property
    def _nulling_baseline(self) -> float:
        if not isinstance(self.nulling_baseline, OptimalNullingBaseline):
            return self.nulling_baseline

        star_habitable_zone_central_angular_radius = self._phringe._scene.star._habitable_zone_central_angular_radius \
            if (self._phringe._scene is not None and self._phringe._scene.star is not None) else None
        return self.nulling_baseline.get_value(
            star_habitable_zone_central_angular_radius,
            self._phringe._instrument.nulling_baseline_min,
            self._phringe._instrument.nulling_baseline_max
        )

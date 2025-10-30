import warnings
from typing import Union, Any

import astropy.units as u
from astropy.units import Quantity
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.io.validation import validate_quantity_units


class OptimalNullingBaseline(BaseModel):
    angular_star_separation: Union[float, str, Quantity]
    wavelength: Union[float, str, Quantity]
    sep_at_max_mod_eff: float

    class Config:
        arbitrary_types_allowed = True

    @field_validator('angular_star_separation')
    def _validate_angular_star_separation(cls, value: Any, info: ValidationInfo) -> Union[float, str]:
        """Validate the angular star separation input.

        :param value: Value given as input
        :param info: ValidationInfo object
        :return: The angular star separation in units of radians or as a string
        """
        if value == 'habitable-zone':
            return value
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.rad,))

    @field_validator('wavelength')
    def _validate_wavelength(cls, value: Any, info: ValidationInfo) -> float:
        """Validate the wavelength input.

        Parameters
        ----------
        value : Any
            Value given as input
        info : ValidationInfo
            ValidationInfo object

        Returns
        -------
        float
            The wavelength in units of length
        """
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,))

    def get_value(
            self,
            star_habitable_zone_central_angular_radius: Union[float, None],
            nulling_baseline_min: float,
            nulling_baseline_max: float
    ) -> float:
        """Compute the optimized nulling baseline.

        :return: The optimized nulling baseline in units of length
        """
        if self.angular_star_separation == 'habitable-zone':
            if star_habitable_zone_central_angular_radius is None:
                raise ValueError(
                    'A star is required to optimize the nulling baseline for the habitable zone. Alternatively, set to an angular value instead of "habitable-zone".')
            angular_star_separation = star_habitable_zone_central_angular_radius
        else:
            angular_star_separation = self.angular_star_separation

        nulling_baseline = self.sep_at_max_mod_eff * self.wavelength / angular_star_separation

        # Set nulling baseline to optimum value or to min/max value if it is outside the allowed range
        if nulling_baseline_min <= nulling_baseline and nulling_baseline <= nulling_baseline_max:
            return nulling_baseline
        elif nulling_baseline < nulling_baseline_max:
            warnings.warn(
                f"Nulling baseline of {nulling_baseline} is below the min allowed baseline of {nulling_baseline_min}. Setting to min baseline.")
            return nulling_baseline_min
        elif nulling_baseline > nulling_baseline_max:
            warnings.warn(
                f"Nulling baseline of {nulling_baseline} is above the max allowed baseline of {nulling_baseline_max}. Setting to max baseline.")
            return nulling_baseline_max

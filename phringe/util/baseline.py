from typing import Union, Any

import astropy.units as u
from astropy.units import Quantity
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo

from phringe.io.validation import validate_quantity_units


class OptimizedNullingBaseline(BaseModel):
    angular_star_separation: Union[float, str, Quantity]
    wavelength: Union[float, str, Quantity]
    separation_at_max_mod_efficiency: float

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

    def get_value(self, star_habitable_zone_central_radius: Union[float, None]) -> float:
        """Compute the optimized nulling baseline.

        :return: The optimized nulling baseline in units of length
        """
        if self.angular_star_separation == 'habitable-zone':
            if star_habitable_zone_central_radius is None:
                raise ValueError(
                    'A star is required to optimize the nulling baseline for the habitable zone. Alternatively, set to an angular value instead of "habitable-zone".')
            angular_star_separation = star_habitable_zone_central_radius
        else:
            angular_star_separation = self.angular_star_separation

        return self.separation_at_max_mod_efficiency * self.wavelength / angular_star_separation

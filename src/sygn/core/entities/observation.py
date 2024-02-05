from typing import Optional, Union, Any

import astropy
from astropy.units import Quantity
from pydantic import BaseModel

from src.sygn.core.entities.base_component import BaseComponent


class Observation(BaseComponent, BaseModel):
    """Class representing the observation."""
    total_integration_time: str
    exposure_time: str
    modulation_period: str
    baseline_ratio: int
    baseline_maximum: str
    baseline_minimum: str
    optimized_differential_output: int
    optimized_star_separation: str
    optimized_wavelength: str

    # def __init__(self):
    #     pass
    def prepare(self):
        """Prepare the observation for the simulation."""
        pass
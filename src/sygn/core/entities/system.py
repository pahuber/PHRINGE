from pydantic import BaseModel

from src.sygn.core.entities.base_component import BaseComponent
from src.sygn.core.entities.photon_sources.exozodi import Exozodi
from src.sygn.core.entities.photon_sources.planet import Planet
from src.sygn.core.entities.photon_sources.star import Star


class System(BaseComponent, BaseModel):
    """Class representing the planetary system."""
    star: Star
    planets: list[Planet]
    zodi: Exozodi

    # def __init__(self):
    #     pass
    def prepare(self, settings, observation, observatory):
        """Prepare the system for the simulation."""
        pass
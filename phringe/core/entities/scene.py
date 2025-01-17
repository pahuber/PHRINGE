from typing import Any

from pydantic import BaseModel

from phringe.core.entities.sources.base_source import BasePhotonSource
from phringe.core.entities.sources.exozodi import Exozodi
from phringe.core.entities.sources.local_zodi import LocalZodi
from phringe.core.entities.sources.planet import Planet
from phringe.core.entities.sources.star import Star


class Scene(BaseModel):
    """Class representing the observation scene.

    :param star: The star in the scene
    :param planets: The planets in the scene
    :param exozodi: The exozodi in the scene
    :param local_zodi: The local zodi in the scene
    """
    star: Star
    planets: list[Planet]
    exozodi: Exozodi
    local_zodi: LocalZodi = None
    _device: Any = None

    # spectrum_list: Any = None
    # maximum_simulation_wavelength_steps: Any = None

    def __init__(self, **data):
        """Constructor method.
        """
        super().__init__(**data)
        self.local_zodi = LocalZodi()
        # self._prepare_unbinned_planets_spectral_flux_densities()

    def get_all_sources(self) -> list[BasePhotonSource]:
        """Return all sources in the scene.

        """
        return [*self.planets, self.star, self.local_zodi, self.exozodi]

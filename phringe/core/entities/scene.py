from typing import Any

from pydantic import BaseModel

from phringe.core.entities.sources.base_source import CachedAttributesSource
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
    star: Star = None
    planets: list[Planet] = []
    exozodi: Exozodi = None
    local_zodi: LocalZodi = None
    _device: Any = None
    _instrument: Any = None
    _grid_size: int = None
    _simulation_time_steps: Any = None

    def add_source(self, source: CachedAttributesSource):
        """Add a source to the scene.

        :param source: The source to add
        """
        source._device = self._device
        source._instrument = self._instrument
        source._grid_size = self._grid_size

        if isinstance(source, Star):
            self.star = source
        elif isinstance(source, Planet):
            source._simulation_time_steps = self._simulation_time_steps
            source.host_star_distance = self.star.distance if source.host_star_distance is None else source.host_star_distance
            source.host_star_mass = self.star.mass if source.host_star_mass is None else source.host_star_mass
            self.planets.append(source)
        elif isinstance(source, Exozodi):
            self.exozodi = source
        elif isinstance(source, LocalZodi):
            self.local_zodi = source

    def remove_source(self, name: str):
        """Remove a source from the scene.

        :param name: The name of the source to remove
        """
        source = self.get_source(name)
        if isinstance(source, Star):
            self.star = None
        elif isinstance(source, Planet):
            self.planets.remove(source)
        elif isinstance(source, Exozodi):
            self.exozodi = None
        elif isinstance(source, LocalZodi):
            self.local_zodi = None

    def get_all_sources(self) -> list[CachedAttributesSource]:
        """Return all sources in the scene.

        """
        all_sources = []
        if self.planets:
            all_sources.extend(self.planets)
        if self.star:
            all_sources.append(self.star)
        if self.local_zodi:
            all_sources.append(self.local_zodi)
        if self.exozodi:
            all_sources.append(self.exozodi)
        return all_sources

    def get_source(self, name: str) -> CachedAttributesSource:
        """Return the source with the given name.

        :param name: The name of the source
        """
        for source in self.get_all_sources():
            if source.name == name:
                return source
            raise ValueError(f'No source with name {name} found in the scene')

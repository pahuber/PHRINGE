from phringe.core.base_entity import BaseEntity
from phringe.core.sources.base_source import BaseSource
from phringe.core.sources.exozodi import Exozodi
from phringe.core.sources.local_zodi import LocalZodi
from phringe.core.sources.planet import Planet
from phringe.core.sources.star import Star


class Scene(BaseEntity):
    """Class representing the observation scene.

    Attributes
    ----------
    star : Star
        The star in the scene
    planets : list[Planet]
        The planets in the scene
    exozodi : Exozodi
        The exozodiacal dust in the scene
    local_zodi : LocalZodi
        The local zodiacal dust in the scene
    """
    star: Star = None
    planets: list[Planet] = []
    exozodi: Exozodi = None
    local_zodi: LocalZodi = None

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "_phringe":
            for source in self._get_all_sources():
                source._phringe = value

    def add_source(self, source: BaseSource):
        """Add a source to the scene.

        Parameters
        ----------
        source : BaseSource
            The source to add
        """
        source._phringe = self._phringe
        if isinstance(source, Star):
            self.star = source
        elif isinstance(source, Planet):
            self.planets.append(source)
        elif isinstance(source, Exozodi):
            self.exozodi = source
        elif isinstance(source, LocalZodi):
            self.local_zodi = source

    def remove_source(self, name: str):
        """Remove a source from the scene.

        Parameters
        ----------
        name : str
            The name of the source to remove
        """
        source = self._get_source(name)
        if isinstance(source, Star):
            self.star = None
        elif isinstance(source, Planet):
            self.planets.remove(source)
        elif isinstance(source, Exozodi):
            self.exozodi = None
        elif isinstance(source, LocalZodi):
            self.local_zodi = None

    def _get_all_sources(self) -> list[BaseSource]:
        """Return all all_sources in the scene.

        """
        all_sources = []
        if self.planets:
            all_sources.extend(self.planets)
        if self.star is not None:
            all_sources.append(self.star)
        if self.local_zodi is not None:
            all_sources.append(self.local_zodi)
        if self.exozodi is not None:
            all_sources.append(self.exozodi)
        return all_sources

    def _get_source(self, name: str) -> BaseSource:
        """Return the source with the given name.

        :param name: The name of the source
        """
        for source in self._get_all_sources():
            if source.name == name:
                return source
        raise ValueError(f'No source with name {name} found in the scene')

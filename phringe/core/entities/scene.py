from pydantic import BaseModel

from phringe.core.entities.base_component import BaseComponent
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.photon_sources.star import Star


class Scene(BaseComponent, BaseModel):
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

    def __init__(self, **data):
        """Constructor method.
        """
        super().__init__(**data)
        self.local_zodi = LocalZodi()

    def get_all_sources(self, has_stellar_leakage: bool, has_local_zodi_leakage: bool, has_exozodi_leakage: bool):
        """Return all sources in the scene that should be accounted for, i.e. only the ones for which a spectrum has
        been generated/provided.

        :param has_stellar_leakage: Whether the star should be accounted for
        :param has_local_zodi_leakage: Whether the local zodi should be accounted for
        :param has_exozodi_leakage: Whether the exozodi should be accounted for
        """
        sources = [*self.planets]
        if has_stellar_leakage:
            sources.append(self.star)
        if has_local_zodi_leakage:
            sources.append(self.local_zodi)
        if has_exozodi_leakage:
            sources.append(self.exozodi)
        return sources

    def prepare(self, settings, observatory, spectrum):
        """Prepare the system for the simulation.

        :param settings: The settings object
        :param observatory: The observatory object
        :param spectrum: The spectrum object
        """
        for planet in self.planets:
            planet.prepare(
                settings.wavelength_steps,
                settings.grid_size,
                time_steps=settings.time_steps,
                has_planet_orbital_motion=settings.has_planet_orbital_motion,
                star_distance=self.star.distance,
                star_mass=self.star.mass,
                number_of_wavelength_steps=len(settings.wavelength_steps)
            )
        if settings.has_stellar_leakage:
            self.star.prepare(
                settings.wavelength_steps,
                settings.grid_size,
                number_of_wavelength_steps=len(settings.wavelength_steps)
            )
        if settings.has_local_zodi_leakage:
            self.local_zodi.prepare(
                settings.wavelength_steps,
                settings.grid_size,
                field_of_view=observatory.field_of_view,
                star_right_ascension=self.star.right_ascension,
                star_declination=self.star.declination,
                number_of_wavelength_steps=len(settings.wavelength_steps)
            )
        if settings.has_exozodi_leakage:
            self.exozodi.prepare(settings.wavelength_steps,
                                 settings.grid_size,
                                 field_of_view=observatory.field_of_view,
                                 star_distance=self.star.distance,
                                 star_luminosity=self.star.luminosity, )

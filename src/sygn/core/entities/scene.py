from pydantic import BaseModel

from src.sygn.core.entities.base_component import BaseComponent
from src.sygn.core.entities.photon_sources.exozodi import Exozodi
from src.sygn.core.entities.photon_sources.local_zodi import LocalZodi
from src.sygn.core.entities.photon_sources.planet import Planet
from src.sygn.core.entities.photon_sources.star import Star


class Scene(BaseComponent, BaseModel):
    """Class representing the observation scene."""
    star: Star
    planets: list[Planet]
    exozodi: Exozodi
    local_zodi: LocalZodi = None

    def __init__(self, **data):
        super().__init__(**data)
        self.local_zodi = LocalZodi()

    def prepare(self, settings, observatory, spectrum):
        """Prepare the system for the simulation."""
        self.star.prepare(
            settings.wavelength_steps,
            settings.grid_size,
            number_of_wavelength_steps=len(settings.wavelength_steps)
        )
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
        self.local_zodi.prepare(
            settings.wavelength_steps,
            settings.grid_size,
            field_of_view=observatory.field_of_view,
            star_right_ascension=self.star.right_ascension,
            star_declination=self.star.declination,
            number_of_wavelength_steps=len(settings.wavelength_steps)
        )
        # self.exozodi.prepare(settings.time_steps, settings.wavelength_steps)

from pydantic import BaseModel

from sygn.core.entities.base_component import BaseComponent
from sygn.core.entities.photon_sources.exozodi import Exozodi
from sygn.core.entities.photon_sources.local_zodi import LocalZodi
from sygn.core.entities.photon_sources.planet import Planet
from sygn.core.entities.photon_sources.star import Star


class Scene(BaseComponent, BaseModel):
    """Class representing the observation scene."""
    star: Star
    planets: list[Planet]
    exozodi: Exozodi
    local_zodi: LocalZodi = None

    def __init__(self, **data):
        super().__init__(**data)
        self.local_zodi = LocalZodi()

    def get_all_sources(self):
        """Return all sources in the scene that should be accounted for, i.e. only the ones for which a spectrum has
        been generated/provided."""
        sources = [*self.planets]
        if self.star.mean_spectral_flux_density is not None:
            sources.append(self.star)
        if self.local_zodi.mean_spectral_flux_density is not None:
            sources.append(self.local_zodi)
        if self.exozodi.mean_spectral_flux_density is not None:
            sources.append(self.exozodi)
        return sources

    def prepare(self, settings, observatory, spectrum):
        """Prepare the system for the simulation."""
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
            pass
            # self.exozodi.prepare(settings.time_steps, settings.wavelength_steps)
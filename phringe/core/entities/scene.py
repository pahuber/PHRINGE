from typing import Any

import numpy as np
import spectres
from astropy import units as u
from pydantic import BaseModel

from phringe.core.entities.base_component import BaseComponent
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.photon_sources.star import Star
from phringe.util.spectrum import _convert_spectrum_units


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
    wavelength_range_lower_limit: Any
    wavelength_range_upper_limit: Any
    local_zodi: LocalZodi = None
    spectrum_list: Any = None
    maximum_simulation_wavelength_steps: Any = None

    def __init__(self, **data):
        """Constructor method.
        """
        super().__init__(**data)
        self.local_zodi = LocalZodi()
        self._prepare_unbinned_planets_spectral_flux_densities()

    def _prepare_unbinned_planets_spectral_flux_densities(self):
        """Calculate the unbinned mean spectral flux densities for all planets. If an input spectrum has been provided,
        crop it to the observatory wavelength range. If no input spectrum has been provided, generate a blackbody
        spectrum.
        """
        self.maximum_simulation_wavelength_steps = np.linspace(
            self.wavelength_range_lower_limit.to(u.um).value,
            self.wavelength_range_upper_limit.to(u.um).value,
            1000
        )
        planet_names_with_spectra = [spectrum_context.planet_name for spectrum_context in
                                     self.spectrum_list] if self.spectrum_list else []
        for planet in self.planets:
            if planet.name in planet_names_with_spectra:
                spectrum_context = next(
                    (spectrum_context for spectrum_context in self.spectrum_list if
                     spectrum_context.planet_name == planet.name),
                    None)
                spectrum_unit = spectrum_context.spectral_flux_density.unit
                new_spec = spectres.spectres(
                    self.maximum_simulation_wavelength_steps,
                    spectrum_context.wavelengths.to(u.um).value,
                    spectrum_context.spectral_flux_density.value,
                    fill=0) * spectrum_unit

                solid_angle = np.pi * (planet.radius.to(u.m) / (self.star.distance.to(u.m)) * u.rad) ** 2
                planet.mean_spectral_flux_density = _convert_spectrum_units(
                    new_spec,
                    self.maximum_simulation_wavelength_steps * u.um,
                    solid_angle.to(u.sr)
                )
            else:
                planet.mean_spectral_flux_density = planet.calculate_blackbody_spectrum(
                    wavelength_steps=self.maximum_simulation_wavelength_steps * u.um,
                    star_distance=self.star.distance
                )

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

    def prepare(self, settings, observation, observatory):
        """Prepare the system for the simulation.

        :param settings: The settings object
        :param observatory: The observatory object
        """
        for planet in self.planets:
            planet.prepare(
                settings.simulation_wavelength_steps,
                settings.grid_size,
                time_steps=settings.simulation_time_steps,
                has_planet_orbital_motion=settings.has_planet_orbital_motion,
                star_distance=self.star.distance,
                star_mass=self.star.mass,
                number_of_wavelength_steps=len(settings.simulation_wavelength_steps),
                maximum_wavelength_steps=self.maximum_simulation_wavelength_steps
            )
        if settings.has_stellar_leakage:
            self.star.prepare(
                settings.simulation_wavelength_steps,
                settings.grid_size,
                number_of_wavelength_steps=len(settings.simulation_wavelength_steps)
            )
        if settings.has_local_zodi_leakage:
            self.local_zodi.prepare(
                settings.simulation_wavelength_steps,
                settings.grid_size,
                field_of_view=observatory.field_of_view,
                star_right_ascension=self.star.right_ascension,
                star_declination=self.star.declination,
                number_of_wavelength_steps=len(settings.simulation_wavelength_steps),
                solar_ecliptic_latitude=observation.solar_ecliptic_latitude
            )
        if settings.has_exozodi_leakage:
            self.exozodi.prepare(settings.simulation_wavelength_steps,
                                 settings.grid_size,
                                 field_of_view=observatory.field_of_view,
                                 star_distance=self.star.distance,
                                 star_luminosity=self.star.luminosity,
                                 grid_size=settings.grid_size)

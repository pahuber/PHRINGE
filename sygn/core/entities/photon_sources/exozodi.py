import numpy as np
from pydantic import BaseModel

from sygn.core.entities.photon_sources.base_photon_source import BasePhotonSource


class Exozodi(BasePhotonSource, BaseModel):
    # TODO: Implement class

    def _calculate_mean_spectral_flux_density(self) -> np.ndarray:
        pass

    def _calculate_sky_brightness_distribution(self, time, wavelength) -> np.ndarray:
        pass

    def _calculate_sky_coordinates(self, time) -> np.ndarray:
        pass

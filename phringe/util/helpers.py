from collections import namedtuple

# Named tuple for coordinates
Coordinates = namedtuple('Coordinates', 'x y')

# Named tuple for a spectrum context object. This object contains the planet name, the spectral flux density and the
# wavelength range of the spectrum.
# SpectrumContext = namedtuple('SpectrumContext', 'planet_name spectral_flux_density wavelengths')

InputSpectrum = namedtuple('InputSpectrum', 'planet_name spectral_flux_density wavelengths')

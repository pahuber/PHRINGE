from typing import Tuple

from astropy import units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic


def get_ecliptic_coordinates(
        star_right_ascension,
        star_declination,
        solar_ecliptic_latitude
) -> Tuple:
    """Return the ecliptic coordinates corresponding to the star position in the sky.

    Parameters
    ----------
    star_right_ascension : float
        The right ascension of the star in units of radians.
    star_declination : float
        The declination of the star in units of radians.
    solar_ecliptic_latitude : float
        The solar ecliptic latitude in units of radians.

    Returns
    -------
    tuple
        Tuple containing the ecliptic latitude and relative ecliptic longitude in units of radians.
    """
    coordinates = SkyCoord(ra=star_right_ascension * u.rad, dec=star_declination * u.rad, frame='icrs')
    coordinates_ecliptic = coordinates.transform_to(GeocentricTrueEcliptic)
    ecliptic_latitude = coordinates_ecliptic.lat.to(u.rad).value
    ecliptic_longitude = coordinates_ecliptic.lon.to(u.rad).value
    relative_ecliptic_longitude = ecliptic_longitude - solar_ecliptic_latitude

    return ecliptic_latitude, relative_ecliptic_longitude

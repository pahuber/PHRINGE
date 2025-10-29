.. _use_custom_spectrum:

Custom Spectrum
===============

`PHRINGE` can automatically generate blackbody spectra for the ``Planet`` objects in the ``Scene``, but also allows
the use of custom spectra. This is specified by the ``input_spectrum`` argument of the ``Planet`` class.

Autogenerate A Blackbody Spectrum
---------------------------------

To automatically create a blackbody spectrum, set ``input_spectrum=None``:

.. code-block:: python

    planet = Planet(
        name='Earth',
        input_spectrum=None,
        # Other arguments
    )

Use A Custom Spectrum
---------------------

To use a custom spectrum, an ``InputSpectrum`` object must be created and set as the ``input_spectrum`` argument of the ``Planet`` class.
An input spectrum can be created from a spectrum file or directly from arrays of SED and wavelengths.

.. note::
    Units equivalent to the following are accepted:
    * SED units: 'W/sr/m2/um', 'W/m2/um', 'W/sr/m2/Hz', 'W/m2/Hz', 'Jy/sr', 'Jy', 'erg/s/sr/cm2/A', 'erg/s/cm2/A', 'erg/s/sr/cm2/Hz', 'erg/s/cm2/Hz'
    * Wavelength units: 'um', 'Hz'

A spectrum file must be a TXT file of the following structure:

.. code-block:: text

    # Content of the file spectrum.txt

    # Wavelength (um)   # Flux (W/sr/m2/um)
    4.000000000e+00     6.3191328e-01
    4.020000000e+00     6.3833316e-01
    4.040100000e+00     6.3448855e-01
    4.060300500e+00     6.5480636e-01
    ...                 ...

This can be loaded into an ``InputSpectrum`` object as follows:

.. code-block:: python

    path_to_file = 'path/to/file.txt'

    input_spectrum = InputSpectrum(
        path_to_file=path_to_file, # Alternatively: None
        sed=None, # Alternatively: NumPy array containing the sed
        wavelengths=None, # Alternatively: NumPy array containing the wavelengths
        sed_units='W/sr/m2/um', # Alternatively: u.W / u.sr / u.m**2 / u.um
        wavelength_units='um' # Alternatively: u.um
        observed_planet_radius=None # Only required if sed_units are not given per solid angle, i.e. /sr
        observed_host_star_distance=None # Only required if sed_units are not given per solid angle, i.e. /sr
        )

    planet = Planet(
        name='Earth',
        input_spectrum=input_spectrum,
        # Other arguments
    )


Generating A Spectrum File With NASA PSG
----------------------------------------
It is also possible to use spectrum files generated with `NASA's Planetary Spectrum Generator (PSG) <https://psg.gsfc.nasa.gov/>`_.
The header and third column that are contained within these files are simply ignored upon importing them into PHRINGE.

.. note::
    When creating spectrum files with PSG the following points should be taken into account:

    * No stellar contribution should be present in the spectrum: Under Change Object > Parent star type, select "None"
    * The distance under Change Object > Distance is irrelevant for the chosen units (the sr part)
    * Under Change Instrument > Telescope / instrument select "User defined"
    * The spectral resolving power should be at least 200 under Change Instrument > Resolution
    * Choose the correct units under Change Instrument > Spectrum intensity unit, i.e. "W/sr/m2/um (spectral radiance)"
    * The spectrum should be independent of a telescope dish size, so under Change Instrument > Beam (FWHM) select "Object-diameter"
    * Under Change Instrument > Noise select "None"

An example spectrum file generated with NASA PSG is shown below:

.. include:: ../_static/spectrum.txt
   :literal:

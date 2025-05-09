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
An input spectrum can be created from a spectrum file or directly from fluxes and wavelengths.

.. note::
    The wavelengths need to be input in units of microns, and the fluxes in units of W/sr/m2/um.

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

    path_to_spectrum = 'path/to/spectrum.txt'

    input_spectrum = InputSpectrum(
        path_to_spectrum=path_to_spectrum, # Alternatively: None
        fluxes=None, # Alternatively: Tensor containing the fluxes in W/sr/m2/um
        wavelengths=None # Alternatively: Tensor containing the wavelengths in micron
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

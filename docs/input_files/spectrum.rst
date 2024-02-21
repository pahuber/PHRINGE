.. _spectrum:

Spectrum File
=============

File Structure
--------------

The spectrum file is a TXT file that contains the input spectrum for a planet. The file must contain two columns, the
first one containing the wavelengths in microns and the second one containing the spectral radiances in W/sr/m2/um. A
typical spectrum file looks like this:

.. include:: ../_static/spectrum.txt
   :literal:

NASA PSG Files
--------------

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

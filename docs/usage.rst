.. _usage:

Usage
=====

Input Files
-----------

PHRINGE requires the following files to run:

* A :ref:`configuration file <configuration>` to configure the simulation settings, observation strategy and observatory hardware
* An :ref:`exoplanetary system file <exoplanetary_system>` to specify the observed exoplanetary system including the stellar, planetary and exozodi properties
* Optional: A :ref:`spectrum file <spectrum>` that represents the spectral radiance in W/sr/m2/um of the planet

.. note::
    If no spectrum file is provided for a planet, a blackbody spectrum will be created from the exoplanetary system file.

Usage From Command Line
------------

PHRINGE can be run from the command line as follows:

.. code-block:: console

    phringe [OPTIONS] CONFIG EXOPLANETARY_SYSTEM [FLAGS]

Arguments
~~~~~~~~~
.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Argument
     - Type
     - Description
   * - ``CONFIG``
     - PATH
     - Path to the configuration file
   * - ``EXOPLANETARY_SYSTEM``
     - PATH
     - Path to the exoplanetary system file

Options
~~~~~~~
.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``-s``, ``--spectrum``
     - <TEXT PATH>
     - Tuple of the planet name as specified in the exoplanetary system file and the path to the corresponding spectrum file; option can be used multiple times for multiplanetary systems
   * - ``-o``, ``--output-dir``
     - PATH
     - Path to the output directory
   * - ``-h``, ``--help``
     - \-
     - Show the help message and exit
   * - ``-v``, ``--version``
     - \-
     - Show the version number and exit





Flags
~~~~~
.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--fits``/``--no-fits``
     - Save the generated data to a FITS file; default is true
   * - ``--copy``/``--no-copy``
     - Create a copy of the configuration and exoplanetary system files in the output directory; default is true

Usage From Python Module
------------------------



Using Input Files
~~~~~~~~~~~~~~~~~
PHRINGE can also be used from within another Python module in the following way:

.. code-block:: python

    from phringe.phringe import PHRINGE
    from pathlib import Path

    spectrum_files = (('Planet Name', Path('path_to_planet_name_spectrum_file')),)

    phringe = PHRINGE()
    phringe.run(
        config_file_path=Path('path_to_config_file'),
        exoplanetary_system_file_path=Path('path_to_exoplanetary_system_file'),
        spectrum_files=spectrum_files,
        output_dir=Path('path_to_output_directory'),
        write_fits=True,
        create_copy=True
    )

.. hint::
    The ``spectrum_files`` **must** be a tuple of planet name/spectrum file path tuples. If only for one planet
    a spectrum file should be provided, then the trailing comma after that planet name/spectrum tuple is essential to
    still make the input a tuple of tuples, i.e. ``(('Planet Name', Path('path_to_planet_name_spectrum_file')),)`` and
    not ``(('Planet Name', Path('path_to_planet_name_spectrum_file')))``.

Using Manually Created Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, instead of using input files to define the `Settings`, `Observatory`, `Observation` and `Scene` objects,
these objects can also be manually created and then passed to `PHRINGE.run(...)`. For example, defining the `Settings`
manually:

.. code-block:: python

    from phringe.core.entities.settings import Settings
    from phringe.phringe import PHRINGE
    from pathlib import Path

    settings = Settings(
        grid_size=20,
        has_planet_orbital_motion=False,
        has_stellar_leakage=False,
        has_local_zodi_leakage=False,
        has_exozodi_leakage=False,
        has_amplitude_perturbations=False,
        has_phase_perturbations=False,
        has_polarization_perturbations=False
    )

    spectrum_files = (('Planet Name', Path('path_to_planet_name_spectrum_file')),)

    phringe = PHRINGE()
    phringe.run(
        config_file_path=Path('path_to_config_file'),
        exoplanetary_system_file_path=Path('path_to_exoplanetary_system_file'),
        settings=settings,
        spectrum_files=spectrum_files,
        output_dir=Path('path_to_output_directory'),
        write_fits=True,
        create_copy=True,
        generate_separate=False
    )

.. note::
    Note that the `settings` object will overwrite the settings defined in the configuration file, if the settings are configured there.

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
   * - ``--stats``/``--no-stats``
     - Save the generated data to a separate FITS file for each photon source to enable photon statistics; default is false

Usage From Python Module
------------------------



Using Input Files
~~~~~~~~~~~~~~~~~~~~~
PHRINGE can also be used from within another Python module in the following way:

.. code-block:: python

    from phringe.phringe import PHRINGE
    from pathlib import Path

    tuple_of_spectra_tuples = (('Planet Name', Path('path_to_planet_name_spectrum_file')),)

    phringe = PHRINGE()
    phringe.run(
        config_file_path=Path('path_to_config_file'),
        exoplanetary_system_file_path=Path('path_to_exoplanetary_system_file'),
        spectrum_tuple=tuple_of_spectra_tuples,
        output_dir=Path('path_to_output_directory'),
        write_fits=True,
        create_copy=True,
        enable_stats=False
    )

.. hint::
    The ``tuple_of_spectra_tuples`` **must** be a tuple of planet name/spectrum file path tuples. If only for one planet
    a spectrum file should be provided, then the trailing comma after that planet name/spectrum tuple is essential to
    still make the input a tuple of tuples, i.e. ``(('Planet Name', Path('path_to_planet_name_spectrum_file')),)`` and
    not ``(('Planet Name', Path('path_to_planet_name_spectrum_file')))``.

Using Dictionaries
~~~~~~~~~~~~~~~~~~
Alternatively, instead of passing the configuration and exoplanetary system file paths, the configuration and
exoplanetary system information can also be passed directly as dictionaries:

.. code-block:: python

    from phringe.phringe import PHRINGE
    from pathlib import Path

    config_dict = {
        'settings': {
        'grid_size: 60,
        ...},
        ...
    }

    exoplanetary_system_dict = {
        'star': {
        'name: 'Sun',
        ...},
        ...
    }

    tuple_of_spectra_tuples = (('Planet Name', Path('path_to_planet_name_spectrum_file')),)

    phringe = PHRINGE()
    phringe.run_with_dict(
        config_dict=config_dict,
        exoplanetary_system_dict=exoplanetary_system_dict,
        spectrum_tuple=tuple_of_spectra_tuples,
        output_dir=Path('path_to_output_directory'),
        write_fits=True,
        create_copy=True,
        enable_stats=False
    )

This skips the file reading step and might be especially useful when generating data within loops, where for each loop
e.g. the planet radius should be updated.
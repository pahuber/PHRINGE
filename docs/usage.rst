.. _usage:

Usage
=====

Input Files
-----------

PHRINGE requires the following files to run:
    * A :ref:`configuration file <configuration>` to configure the simulation settings, observation strategy and observatory hardware
    * An :ref:`exoplanetary system file <exoplanetary_system>` to specify the observed exoplanetary system including the stellar, planetary and exozodi properties
    * Optional: A spectrum file that represents the flux of the planet

.. note::
    If no spectrum file is provided, a blackbody spectrum will be created from the exoplanetary system file.

Usage From Command Line
------------

PHRINGE can be run from the command line as follows:

.. code-block:: console

    phringe [OPTIONS] CONFIG_FILE_PATH EXOPLANETARY_SYSTEM_FILE_PATH [FLAGS]

Arguments
~~~~~~~~~
* ``CONFIG_FILE_PATH``: The path to the configuration file
* ``EXOPLANETARY_SYSTEM_FILE_PATH``: The path to the exoplanetary system file

Options
~~~~~~~
* ``-s``, ``--spectrum-file``:   The path to the spectrum file
* ``-o``, ``--output-dir``:     The path to the output directory
* ``-h``, ``--help``:            Show the help message and exit
* ``-v``, ``--version``:         Show the version number and exit

Flags
~~~~~
* ``--fits``/``--no-fits``:      Save the generated data to a FITS file; default is true
* ``--copy``/``--no-copy``       Create a copy of the configuration and exoplanetary system files in the output directory; default is true


Usage From Python Module
------------------------



Using Input Files
~~~~~~~~~~~~~~~~~
PHRINGE can also be used from within another Python module by making use of its API in the following way:

.. code-block:: python

    from phringe import API
    from pathlib import Path

    data = API.generate_data(
        Path('path_to_config_file'),
        Path('path_to_exoplanetary_system_file'),
        Path('path_to_spectrum_file'),
        output_dir=Path('path_to_output_directory'),
        fits=True,
        copy=True
)

Alternatively, instead of passing the configuration and exoplanetary system file paths, the configuration and
exoplanetary system information can also be passed directly as dictionaries:

.. code-block:: python

    from phringe import API
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

    data = API.generate_data(
        config_dict,
        exoplanetary_system_dict,
        Path('path_to_spectrum_file'),
        output_dir=Path('path_to_output_directory'),
        fits=True,
        copy=True
    )

.. note::
    The latter option might be especially useful when generating data within loops, such that in each iteration a different
    dictionary can be used rather than having to use different files each time.
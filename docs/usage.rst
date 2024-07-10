.. _usage:

Usage
=====

.. note::
    It is recommended to run `PHRINGE` on a GPU, as the simulation gets computationally expensive quickly.

Required User Input
-------------------

`PHRINGE` requires certain user input to configure the `simulation settings`, `observation strategy`, `observatory hardware` and `astrophysical scene`. Usually, this input is provided through the following files:

* A :ref:`configuration file <configuration>` to configure the simulation settings, observation strategy and observatory hardware
* An :ref:`exoplanetary system file <exoplanetary_system>` to specify the observed exoplanetary system including the stellar, planetary and exozodi properties
* Optional: A :ref:`spectrum file <spectrum>` that contains the spectra of the planets in the system in W/sr/m2/um

.. note::
    If no spectrum is provided for a planet, a blackbody spectrum will be created from the planetary properties specified in the exoplanetary system file.

Using Within Python Module
--------------------------
`PHRINGE` can be used from within another Python module by using the ``PHRINGE`` class from the ``phringe.phringe_ui`` module:

.. code-block:: python

    from phringe.phringe_ui import PHRINGE
    from pathlib import Path


    phringe = PHRINGE()
    phringe.run(
        config_file_path=Path('path_to_config_file'),
        exoplanetary_system_file_path=Path('path_to_exoplanetary_system_file')
    )

The input can either be given through input files (as done here; see :doc:`basic example <tutorials/example_basic>`) or by manually creating the required objects (see :doc:`advanced example <tutorials/example_advanced>`).

Using Command Line Interface (CLI)
-----------------------------------

`PHRINGE` features a command line interface (CLI) and can be run from the command line as follows to directly generate FITS files containing the synthetic data:

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
   * - ``-g``, ``--gpus``
     - INTEGER
     - Indices if the GPUs to use; option can be used multiple times for usage of multiple GPUs
   * - ``-o``, ``--output-dir``
     - PATH
     - Path to the output directory
   * - ``-f``, ``--fits-suffix``
     - TEXT
     - Suffix for the FITS file name
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
   * - ``--detailed``/``--no-detailed``
     - Run in detailed mode; default is false
   * - ``--fits``/``--no-fits``
     - Save the generated data to a FITS file; default is true
   * - ``--copy``/``--no-copy``
     - Create a copy of the configuration and exoplanetary system files in the output directory; default is true
   * - ``--dir``/``--no-dir``
     - Create a new directory in the output directory for each run; default is true
   * - ``--normalize``/``--no-normalize``
     - Whether to normalize the data to unit RMS along the time axis; default is false




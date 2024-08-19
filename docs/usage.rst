.. _usage:

Usage
=====

.. note::
    It is recommended to run `PHRINGE` on a GPU, as the simulation gets computationally expensive quickly.

Required User Input
-------------------

`PHRINGE` requires a :ref:`configuration file <configuration>` to configure the simulation, the observation mode, the instrument and the scene.
Alternatively, the input can also be given manually by creating the required objects (see :doc:`advanced example <tutorials/example_advanced>`).


Using Within Python Module
--------------------------
`PHRINGE` can be used from within another Python module by using the ``PHRINGE`` class from the ``phringe.phringe_ui`` module:

.. code-block:: python

    from phringe.phringe_ui import PHRINGE
    from pathlib import Path


    phringe = PHRINGE()
    phringe.run(
        config_file_path=Path('path_to_config_file'),
    )

Alternatively to using configuration files (as done here; see :doc:`basic example <tutorials/example_basic>`), the input can also be manually given by creating the required objects (see :doc:`advanced example <tutorials/example_advanced>`).

Using Command Line Interface (CLI)
-----------------------------------

`PHRINGE` features a command line interface (CLI) and can be run from the command line as follows to directly generate FITS files containing the synthetic data:

.. code-block:: console

    phringe [OPTIONS] CONFIG [FLAGS]

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
   * - ``--fits``/``--no-fits``
     - Save the generated data to a FITS file; default is true
   * - ``--copy``/``--no-copy``
     - Create a copy of the configuration and exoplanetary system files in the output directory; default is true
   * - ``--dir``/``--no-dir``
     - Create a new directory in the output directory for each run; default is true
   * - ``--normalize``/``--no-normalize``
     - Whether to normalize the data to unit RMS along the time axis; default is false




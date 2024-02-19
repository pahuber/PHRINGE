.. _usage:

Usage
=====

Input Files
-----------

SYGN requires the following files to run:
    * A :ref:`configuration file <input_files/configuration>` to configure the simulation settings, observation strategy and observatory hardware
    * An :ref:`exoplanetary system file <input_files/exoplanetary_system>` to specify the observed exoplanetary system including the stellar, planetary and exozodi properties
    * Optional: A spectrum file that represents the flux of the planet

.. note::
    If no spectrum file is provided, a blackbody spectrum will be created from the exoplanetary system file.

Running SYGN From the Command Line
------------

SYGN can be run from the command line as follows:

.. code-block:: console

    sygn [-s SPECTRUM_PATH -o OUTPUT_DIRECTORY_PATH] CONFIG_FILE_PATH EXOPLANETARY_SYSTEM_FILE_PATH

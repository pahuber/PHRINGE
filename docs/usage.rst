.. _usage:

Usage
=====

Input Files
-----------

SYGN requires the following files to run:
    * Configuration file
    * Exoplanetary system file
    * Spectrum file (optional)

.. note::
    If no spectrum file is provided, a blackbody spectrum will be created from the exoplanetary system file.

Configuration File
~~~~~~~~~~~~~~~~~~

The configuration file is a YAML file that is used to configure the simulation settings, the observation strategy and
the observatory hardware. See the :ref:`user documentation <configuration>` for more information.

A typical configuration file looks like this:

.. include:: _static/config.yaml
   :literal:

Exoplanetary System File
~~~~~~~~~~~~~~~~~~~~~

The exoplanetary system file is a YAML file that contains information about the observed exoplanetary system. See the
:ref:`user documentation <exoplanetary_system>` for more information.


A typical exoplanetary system file looks like this:

.. include:: _static/exoplanetary_system.yaml
   :literal:

Spectrum File
~~~~~~~~~~~~

The spectrum file is a TXT file and contains the planet spectrum. It should contain two columns, the first column
containing the wavelength in microns and the second column containing the flux in W/m^2/micron.


Running SYGN From the Command Line
------------

SYGN can be run from the command line as follows:

.. code-block:: console

    sygn [-s PATH_TO_SPECTRUM_FILE -o PATH_TO_OUTPUT_DIRECTORY] PATH_TO_CONFIG_FILE PATH_TO_EXOPLANETARY_SYSTEM_FILE
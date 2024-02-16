.. _usage:

Usage
=====

Input Files
-----------

SYGN requires the following files to run:
    * Configuration file
    * Spectrum context file
    * Spectrum file (optional)

If no spectrum file is provided, a blackbody spectrum will be created from the spectrum context file.

Configuration File
~~~~~~~~~~~~~~~~~~

The configuration file is a YAML file that is used to configure the simulation settings, the observation strategy and
the observatory hardware. See :ref:`user documentation <configuration>` for more information.

A typical configuration file looks like this:

.. include:: _static/config.yaml
   :literal:

Spectrum Context File
~~~~~~~~~~~~~~~~~~~~~

The spectrum context file is a YAML file that contains important context information about the input planet spectrum. See :ref:`user documentation <spectrum_context>` for more information.


A typical spectrum context file looks like this:

.. include:: _static/spectrum_context.yaml
   :literal:

Spectrum File
~~~~~~~~~~~~

The spectrum file is a TXT file and contains the planet spectrum. It should contain two columns, the first column
containing the wavelength in microns and the second column containing the flux in W/m^2/micron.


Running SYGN
------------

SYGN can be run from the terminal as follows:

.. code-block:: console

    sygn [-s PATH_TO_SPECTRUM_FILE -o PATH_TO_OUTPUT_DIRECTORY] PATH_TO_CONFIG_FILE PATH_TO_SPECTRUM_CONTEXT_FILE
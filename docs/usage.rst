.. _usage:

Usage
=====

Prerequisites
-----------

SYGN requires a configuration file and a spectrum context file to run. Optionally, a spectrum file containing a planet
spectrum can be provided in addition to the two required files. If no spectrum file is provided, a blackbody spectrum
will be created from the spectrum context file.

Configuration File
~~~~~~~~~~~~~~~~~~

The configuration file is a YAML file that is used to configure the simulation settings, the observation strategy and
the observatory hardware. A typical configuration file looks like this:

.. include:: _static/config.yaml
   :literal:

Spectrum Context File
~~~~~~~~~~~~~~~~~~~~~

The spectrum context file is a YAML file that contains important context information about the input planet spectrum.
A typical spectrum context file looks like this:

.. include:: _static/spectrum_context.yaml
   :literal:


Running SYGN
------------

Once installed SYGN can be used from the terminal as follows:

.. code-block:: console

    sygn [-s PATH_TO_SPECTRUM_FILE -o PATH_TO_OUTPUT_DIRECTORY] PATH_TO_CONFIG_FILE PATH_TO_SPECTRUM_CONTEXT_FILE
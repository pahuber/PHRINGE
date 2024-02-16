.. first_example:

First Example
=============

This example demonstrates how to use SYGN to generate synthetic photometry data for an Earth twin orbiting a Sun twin
at 10 pc.

Download the Configuration File
--------------------------------

Open a terminal and enter the following command to download the configuration file:

.. code-block:: console

    wget -O config.yaml https://raw.githubusercontent.com/pahuber/SYGN/main/examples/config.yaml


Download the Spectrum Context File
----------------------------------

In the terminal, enter the following command to download the spectrum context file:

.. code-block:: console

    wget -O spectrum_context.yaml https://raw.githubusercontent.com/pahuber/SYGN/main/examples/system.yaml

Run SYGN
--------

To run SYGN enter the following command in the terminal:

.. code-block:: console

    sygn config.yaml spectrum_context.yaml


Output
------

This will generate a FITS file called `data_{timestamp}.fits` containing the synthetic photometry data. Opening the file
in a FITS viewer will reveal the following image:

.. image:: _static/first_example.jpg
    :alt: First Example
    :width: 100%

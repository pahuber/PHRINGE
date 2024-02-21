.. first_example:

First Example
=============

This example demonstrates how to use PHRINGE to generate synthetic photometry data for an Earth twin orbiting a Sun twin
at 10 pc.

Download the Configuration File
--------------------------------

Open a terminal and enter the following command to download the configuration file:

.. code-block:: console

    wget -O config.yaml https://raw.githubusercontent.com/pahuber/PHRINGE/main/docs/_static/config.yaml


Download the Exoplanetary System File
----------------------------------

In the terminal, enter the following command to download the exoplanetary system file:

.. code-block:: console

    wget -O exoplanetary_system.yaml https://raw.githubusercontent.com/pahuber/PHRINGE/main/docs/_static/exoplanetary_system.yaml

Run PHRINGE
--------

To generate the synthetic data run the following command in the terminal:

.. code-block:: console

    phringe config.yaml exoplanetary_system.yaml


Output
------

This will generate a FITS file called `data_{timestamp}.fits` containing the synthetic photometry data. Opening the file
in a FITS viewer will reveal (due to the randomness involved) something similar to the following image:

.. image:: _static/first_example.jpg
    :alt: First Example
    :width: 100%

Here, the brightness corresponds to the photon counts, while the x-axis corresponds to time and the y-axis to wavelength
/spectral channel.
.. first_example:

First Example
=============


To generate synthetic photometry data with SYGN, a configuration file and a spectrum context file is required. To
download an example configuration and spectrum context file open a terminal and enter:

.. code-block:: console

    wget -O config.yaml https://raw.githubusercontent.com/pahuber/SYGN/main/examples/config.yaml
    wget -O spectrum_context.yaml https://raw.githubusercontent.com/pahuber/SYGN/main/examples/system.yaml

Then run:

.. code-block:: console

    sygn config.yaml spectrum_context.yaml

This will generate a FITS file called `data_{timestamp}.fits` containing the synthetic photometry data.

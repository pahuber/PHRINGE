.. _first_example:

First Example
=============

This example illustrates the basic usage of *PHRINGE* with a config file (`download config file here <https://github.com/pahuber/PHRINGE/blob/main/docs/_static/first_example.ipynb>`_).
For more advanced use cases, see the `Advanced Example <advanced_example.ipynb>`_.

In the following, we will generate and plot a synthetic data set for a double Bracewell nuller in the Emma-X configuration observing an Earth-like exoplanet (with a blackbody spectrum) around a Sun twin at 10 pc.

To create your own instrument or use a custom planet spectrum, have a look at the `Custom Instrument <create_custom_instrument.rst>`_ and `Custom Spectrum <use_custom_spectrum.rst>`_ tutorials.

Config File
~~~~~~~~~~~

We use the config file to specify the observation, instrument, and astrophysical scene:

.. literalinclude:: ../_static/config.py
   :language: python


Python Script
~~~~~~~~~~~~~

.. include:: ../_static/first_example.ipynb
   :parser: myst_nb.docutils_

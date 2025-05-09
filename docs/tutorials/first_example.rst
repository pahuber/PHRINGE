.. _first_example:

First Example
=============

The easiest way to run *PHRINGE* is by the use of a config file. More advanced use cases might require more control over the input, which is possible by manually setting up the objects specified in the config file (see `Advanced Example <advanced_example.ipynb>`_).

In this first example, we simulate the observation of an Earth twin around a Sun twin at 10 pc with a basic X-Array Double Bracewell nuller.

Config File
~~~~~~~~~~~

We use the following config file saved in ``../_static/config.py`` to specify the observation, instrument, and astrophysical scene:

.. literalinclude:: ../_static/config.py
   :language: python


Python Script
~~~~~~~~~~~~~

.. include:: ../_static/first_example.ipynb
   :parser: myst_nb.docutils_

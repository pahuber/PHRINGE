.. _create_config_file:

Creating A Config File
======================

A ``Configuration`` object requires either a path to a configuration file containing the configuration dictionary or the configuration dictionary itself:

.. code-block:: python

    # Specify the path to the configuration file
    config = Configuration(path='path/to/config.py')

    # Specify the configuration dictionary directly
    config = Configuration(config_dict=config_dict)

The following shows an example configuration file with the required structure. For information about the individual keys
and their possible values, see the :ref:`api_documentation` section.

.. include:: ../_static/config.py
   :literal:


.. _usage:

Usage
=====
A `PHRINGE` script starts with an instance of the :doc:`PHRINGE <source/phringe>` class. Upon its initialization, the user
has to specify the observation, instrument and astrophysical scene parameters. This can be done in two ways, as described below.

.. note::
    It is recommended to run `PHRINGE` on a GPU, as the simulation gets computationally expensive quickly and may take a substantial amount of time on CPUs.

1. Using Config Files (Recommended)
-----------------------------------

The recommended way to set up a `PHRINGE` simulation is to use a `config file <_static/config.py>`_ together with a :doc:`Configuration <source/configuration>` object (see `First Example <tutorials/first_example_a.rst>`_).
The following example shows how to calculate the detector counts using this approach:


.. code-block:: python

    # Create a PHRINGE object
    phringe = PHRINGE()

    # Get objects from a config file
    config = Configuration(path="path/to/config.py")
    phringe.set(config)

    # Calculate counts
    counts = phringe.get_counts()


2. Manually Creating Objects (Advanced)
---------------------------------------

Alternatively, one can manually create the :doc:`Observation <source/observation>`, :doc:`Instrument <source/instrument>` and :doc:`Scene <source/scene>` objects (see `Advanced Example <tutorials/advanced_example_a.rst>`_).
This might be required for more advanced use cases such as looping through a parameter space.
The following example shows how to calculate the detector counts using this approach:

.. code-block:: python

    # Create a PHRINGE object
    phringe = PHRINGE()

    # Create objects manually
    obs = Observation(...) # Arguments omitted here for brevity
    phringe.set(obs) # This will overwrite the the observation if defined in the config file

    inst = Instrument(...) # Arguments omitted here for brevity
    phringe.set(inst) # This will overwrite the instrument if defined in the config file

    scene = Scene(...) # Arguments omitted here for brevity
    phringe.set(scene) # This will overwrite the scene if defined in the config file

    # Calculate counts
    counts = phringe.get_counts()





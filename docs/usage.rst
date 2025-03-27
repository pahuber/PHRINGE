.. _usage:

Usage
=====
At the beginning of every simulation done with `PHRINGE` is the ``PHRINGE`` class. Upon its initialization, the user has to set all other required objects, before then calculating the detector counts. The following code snippet gives a quick overview of how to use `PHRINGE`:

.. code-block:: python

    # Create a PHRINGE object
    phringe = PHRINGE()

    # Set other objects
    phringe.set(...)
    phringe.set(...)
    phringe.set(...)

    # Calculate counts
    counts = phringe.get_counts()


The ``PHRINGE`` object is responsible for the heavy computations and provides an interface for the user to retrieve the generated data (see :doc:`PHRINGE documentation <source/phringe>`). To calculate the detector counts, it requires information about the `observation`, the `instrument`, and the astrophysical `scene`.
There are two ways to provide this information.

Option 1: Using a Config File
----------------------------

The simplest way to use `PHRINGE` is by using a config file and a ``Configuration`` object (see :doc:`Configuration documentation <source/configuration>`):

.. code-block:: python

    config = Configuration.from_file("path/to/config.yaml")
    phringe.set(config)

Option 2: Manually Creating Objects
-----------------------------------

Alternatively, one can manually set the objects (see :doc:`Observation documentation <source/observation>`, :doc:`Instrument documentation <source/instrument>` or :doc:`Scene documentation <source/scene>`):

.. code-block:: python

    obs = Observation(...)
    inst = Instrument(...)
    scene = Scene(...)

    phringe.set(obs)
    phringe.set(inst)
    phringe.set(scene)

This may be required for more advanced use cases.

.. note::
    It is recommended to run `PHRINGE` on a GPU, as the simulation gets computationally expensive quickly and may take a substantial amount of time on CPUs.

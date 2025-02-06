.. _usage:

Usage
=====
`PHRINGE` can be used by creating a

*  ``PHRINGE``

object. This object is resposible for the heavy computations and provides an interface for the user to retrieve the generated data (see :doc:`PHRINGE documentation <source/phringe>`).
It requires the following three objects for its setup:

*  ``Observation``,
*  ``Instrument``,
*  ``Scene``.

These objects hold the necessary information for the calculations (see :doc:`Observation documentation <source/observation>`, :doc:`Instrument documentation <source/instrument>` or :doc:`Scene documentation <source/scene>`). They can be created manually, or, alternatively, be set up automatically using a configuration dictionary/file and a

*  ``Configuration``

object (see :doc:`Configuration documentation <source/configuration>`).

.. note::
    It is recommended to run `PHRINGE` on a GPU, as the simulation gets computationally expensive quickly and may take a substantial amount of time on CPUs.

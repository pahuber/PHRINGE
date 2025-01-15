.. _index:

Welcome to the PHRINGE Docs
===========================

**PHRINGE** is a **PH**\ otoelectron counts generato\ **R** for null\ **ING** int\ **E**\ rferometers capable of
generating synthetic data for space-based nulling interferometers observing exoplanetary systems. It is being developed
in the context of the `Large Interferometer For Exoplanets (LIFE) <https://www.life-space-mission.com>`_ mission and
used within `LIFEsimMC <https://www.github.com/pahuber/lifesimmc>`_. `PHRINGE` is written in Python
and offers the following features:

* Symbolic input of complex amplitude transfer matrix and array positions, ensuring maximum flexibility in instrument architecture modeling
* Symbolic calculation of instrument intensity response
* Noise models for astrophysical noise sources including stellar, local zodi and exozodi leakage
* Noise models for instrumental perturbations including amplitude, phase (OPD) and polarization rotation perturbations
* Export of synthetic data as a NIFITS file, integrating with the `nifits <https://www.github.com/rlaugier/nifits>`_ package



.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   usage
   first_examples

.. toctree::
   :maxdepth: 2
   :caption: User Documentation
   :hidden:

   input_files
   architecture
   api_documentation


.. toctree::
   :maxdepth: 2
   :caption: About
   :hidden:

   about

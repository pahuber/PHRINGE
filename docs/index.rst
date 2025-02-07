.. _index:

Welcome to the PHRINGE Docs
===========================

**PHRINGE** is a **PH**\ otoelectron counts generato\ **R** for null\ **ING** int\ **E**\ rferometers and offers
GPU-accelerated generation of synthetic data for space-based nulling interferometers observing exoplanetary systems.

Features
--------

* | **Flexible Instrument Architecture Modeling:**
  | Support for symbolic input of beam combiner matrices and collector positions thanks to the integration of `SymPy <https://www.sympy.org>`_.
* | **Detailed Instrumental Noise Modeling:**
  | Monte Carlo sampled instrument perturbations such as amplitude, phase and polarization rotation errors.
* | **Custom Astrophysical Scenes:**
  | User-specifiable exoplanetary systems with the option for modeling planetary orbital motion.
* | **Fast Computations:**
  | Option for GPU-accelerated computations thanks to the integration of `PyTorch <https://pytorch.org>`_. It also be run on CPUs.
* | **Support for NIFITS Data Standard:**
  | Option to export synthetic data sets as NIFITS files thanks to the integration of `nifits <https://www.github.com/rlaugier/nifits>`_.

Contact
-------

For questions or other inquiries, please contact Philipp A. Huber (huberph@phys.ethz.ch).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   usage
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Documentation
   :hidden:

   architecture
   api_documentation


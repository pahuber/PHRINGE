.. _index:

Welcome to the PHRINGE Docs
===========================

**PHRINGE** is a **PH**\ otoelectron counts generato\ **R** for null\ **ING** int\ **E**\ rferometers. It offers
GPU-accelerated generation of synthetic data for space-based nulling interferometers observing exoplanetary systems.
`PHRINGE` is written in Python and has been developed in the context of the `Large Interferometer For Exoplanets (LIFE) <https://www.life-space-mission.com>`_
mission and is used within `LIFEsimMC <https://www.github.com/pahuber/lifesimmc>`_.

Features
--------

| **Flexible Instrument Architecture Modeling:**
| Support for symbolic input of beam combiner matrices and collector positions thanks to the integration of `sympy <https://www.sympy.org>`_.

| **Detailed Instrumental Noise Modeling:**
| Monte Carlo sampled instrument perturbations such as amplitude, phase and polarization rotation errors.

| **Custom Astrophysical Scenes:**
| User-specifiable exoplanetary systems with the option for modeling planetary orbital motion.

| **Fast Computations:**
| Option for GPU-accelerated computations thanks to the integration of `PyTorch <https://pytorch.org>`_.

| **Support for NIFITS Data Standard:**
| Option to export synthetic data sets as NIFITS files thanks to the integration of `nifits <https://www.github.com/rlaugier/nifits>`_.

Contact
-------

For questions or other inquiries, please contact the main developer Philipp A. Huber (huberph@phys.ethz.ch).

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


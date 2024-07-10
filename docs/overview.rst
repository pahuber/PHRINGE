.. _overview:

Overview
========

`PHRINGE` is a Python-based tool to generate synthetic photometry data for nulling interferometers. It is being developed in the context of the `Large Interferometer For Exoplanets (LIFE) <https://www.life-space-mission.com>`_ and used for the data generation part of `LIFEsimMC <https://www.github.com/pahuber/lifesimmc>`_.

Features
--------

`PHRINGE` includes the following features:

* Model different array architectures including different array configurations (X-Array, Triangle, Pentagon) and different beam combination schemes (double Bracewell, Kernel)
* Model noise contributions from astrophysical sources including stellar, local zodi and exozodi leakage
* Model noise contributions from instrumental perturbations including amplitude, phase (OPD) and polarization rotation perturbations
* Configure the observation strategy and the observatory hardware with all major parameters
* Configure the observed exoplanetary system including the star, planets and exozodi
* Export the synthetic photometry data as a FITS file
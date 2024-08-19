.. _overview:

Overview
========

`PHRINGE` is a Python-based tool to generate synthetic (photoelectron counts) data for nulling interferometers. It is being developed in the context of the `Large Interferometer For Exoplanets (LIFE) <https://www.life-space-mission.com>`_ and used for the data generation part of `LIFEsimMC <https://www.github.com/pahuber/lifesimmc>`_.

Features
--------

`PHRINGE` includes the following features:

* Symbolic input of complex amplitude transfer matrix and array positions, ensuring maximum flexibility in architecture modeling
* Symbolic calculation of instrument intensity response
* Noise models for astrophysical noise sources including stellar, local zodi and exozodi leakage
* Noise models for instrumental perturbations including amplitude, phase (OPD) and polarization rotation perturbations
* Export of synthetic data as a FITS file
.. _configuration:

Configuration File
==================

Overview
--------

The configuration file is a YAML file that is used to configure the simulation settings, the observation strategy and
the observatory hardware. A typical configuration file looks like this:

.. include:: _static/config.yaml
   :literal:

Note that the file has three main keys: `settings`, `observation` and `observatory`.

Keys
----

Settings
~~~~~~~~~~~~~~~~~~~
The *settings* key contains several other keys that are used to configure the simulation. The key names, value types,
accepted values and meanings are given in the table below:

.. list-table:: Settings Keys
   :widths: 30 10 20 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Description
   * - grid_size
     - int
     - e.g. 60
     - Grid size used for the calculations
   * - has_planet_orbital_motion
     - bool
     - true, false
     - Whether planet orbital motion is modeled
   * - has_stellar_leakage
     - bool
     - true, false
     - Whether stellar leakage is modeled
   * - has_local_zodi_leakage
     - bool
     - true, false
     - Whether local zodi leakage is modeled
   * - has_exozodi_leakage
     - bool
     - true, false
     - Whether exozodi leakage is modeled
   * - has_amplitude_perturbations
     - bool
     - true, false
     - Whether amplitude perturbations are modeled
   * - has_phase_perturbations
     - bool
     - true, false
     - Whether phase perturbations are modeled
   * - has_polarization_perturbations
     - bool
     - true, false
     - Whether polarization perturbations are modeled


Observation
~~~~~~~~~~~~~~~~~~~
The *observation* key contains several other keys that are used to configure the observation strategy. The key names, value types,
accepted values and meanings are given in the table below:

.. list-table:: Observation Keys
   :widths: 30 10 20 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Description
   * - total_integration_time
     - str*
     - e.g. 1 d
     - Total integration time of the observation
   * - exposure_time
     - str*
     - e.g. 0.01 d
     - Exposure/detector integration time
   * - modulation_period
     - str*
     - e.g. 1 d
     - Duration to complete a modulation cycle of the array
   * - baseline_ratio
     - int
     - e.g. 6
     - Ratio of the imaging and the nulling baselines
   * - baseline_maximum
     - str*
     - e.g. 600 m
     - Maximum allowed baseline length
   * - baseline_minimum
     - str*
     - e.g. 10 m
     - Minimum allowed baseline length
   * - optimized_differential_output
     - int
     - e.g. 0
     - Index corresponding to the :math:`n`\ th differential output of the array that the baselines should be optimized for
   * - optimized_star_separation
     - str/str*
     - habitable-zone, e.g. 0.1 arcsec
     - Angular separation between the star and the planet that the baselines should be optimized for; 'habitable-zone' is also a valid input and will optimize for the habitable zone of the star
   * - optimized_wavelength
     - str*
     - e.g. 10 um
     - Wavelength that the baselines should be optimized for

\* String consisting of a number and a unit that can be parsed by astropy.units






Observatory
~~~~~~~~~~~~~~~~~~~
The *observatory* key contains several other keys that are used to configure the observatory hardware. The key names, value types,
accepted values and meanings are given in the table below:

.. list-table:: Observatory Keys
   :widths: 30 10 20 40
   :header-rows: 1

   * - Name
     - Type
     - Description
     - Accepted Values
   * - array_configuration
     - str
     - emma-x-circular-rotation, equilateral-triangle-circular-rotation, regular-pentagon-circular-rotation
     - Array configuration type
   * - beam_combination_scheme
     - str
     - double-bracewell, kernel-3, kernel-4, kernel-5
     - Beam combination scheme; must be compatible with the array configuration type
   * - aperture_diameter
     - str*
     - e.g. 2 m
     - Aperture diameter of the collectors
   * - spectral_resolving_power
     - int
     - e.g. 100
     - Spectral resolving power of the instrument
   * - wavelength_range_lower_limit
     - str*
     - e.g. 4 um
     - Lower limit of the wavelength range
   * - wavelength_range_upper_limit
     - str*
     - e.g. 18 um
     - Upper limit of the wavelength range
   * - unperturbed_instrument_throughput
     - float
     - e.g. 0.1
     - Throughput of the unperturbed instrument
   * - amplitude_perturbation_rms
     - float
     - e.g. 0.7
     - RMS of the amplitude perturbation
   * - amplitude_falloff_exponent
     - int
     - e.g. 1
     - Falloff exponent of the amplitude perturbation power spectrum
   * - phase_perturbation_rms
     - str*
     - e.g. 1 nm
     - RMS of the phase perturbation
   * - phase_falloff_exponent
     - int
     - e.g. 1
     - Falloff exponent of the phase perturbation power spectrum
   * - polarization_perturbation_rms
     - str*
     - e.g. 1 rad
     - RMS of the polarization perturbation
   * - polarization_falloff_exponent
     - int
     - e.g. 1
     - Falloff exponent of the polarization perturbation power spectrum

\* String consisting of a number and a unit that can be parsed by astropy.units

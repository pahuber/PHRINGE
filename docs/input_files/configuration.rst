.. _configuration:

Configuration File
==================

File Structure
--------------

The configuration file is a Python file that is used to configure the simulation settings, the observation mode, the instrument and the scene.
A typical configuration file looks like this:

.. include:: ../_static/config.py
   :literal:

First, the array configuration matrix and the complex amplitude transfer matrix are defined using ``sympy`` symbolic math.
Then, all parameters requried for the data generation are specified in the ``config`` dictionary. Note that the four main keys:
``simulation``, ``observation_mode`` ``instrument`` and ``scene``. A more detailed description of all parameters is given below.


Dictionary Keys
---------------

Simulation
~~~~~~~~~~~~~~~~~~~
The ``simulation`` key contains several other keys that are used to configure the simulation. The key names, value types,
accepted values and meanings are given in the table below:

.. list-table:: Simulation Keys
   :widths: 26 7 20 7 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Dimension
     - Description
   * - grid_size
     - int
     - e.g. 60
     - 1
     - Grid size used for the calculations
   * - time_step_size
     - str*
     - e.g. '1 s'
     - Time
     - Time step size used for the calculations
   * - has_planet_orbital_motion
     - bool
     - true, false
     - \-
     - Whether planet orbital motion is modeled
   * - has_planet_signal
     - bool
     - true, false
     - \-
     - Whether planet signal is modeled
   * - has_stellar_leakage
     - bool
     - true, false
     - \-
     - Whether stellar leakage is modeled
   * - has_local_zodi_leakage
     - bool
     - true, false
     - \-
     - Whether local zodi leakage is modeled
   * - has_exozodi_leakage
     - bool
     - true, false
     - \-
     - Whether exozodi leakage is modeled
   * - has_amplitude_perturbations
     - bool
     - true, false
     - \-
     - Whether amplitude perturbations are modeled
   * - has_phase_perturbations
     - bool
     - true, false
     - \-
     - Whether phase perturbations are modeled
   * - has_polarization_perturbations
     - bool
     - true, false
     - \-
     - Whether polarization perturbations are modeled

\* String consisting of a number and a unit that can be parsed by astropy.units

Observation Mode
~~~~~~~~~~~~~~~~
The ``observation_mode`` key contains several other keys that are used to configure the observation mode. The key names, value types,
accepted values and meanings are given in the table below:

.. list-table:: Observation Mode Keys
   :widths: 26 7 20 7 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Dimension
     - Description
   * - solar_ecliptic_latitude
     - str*
     - e.g. '0 deg'
     - Angle
     - Solar ecliptic latitude used to calculate the contribution of the local zodiacal light
   * - total_integration_time
     - str*
     - e.g. '1 d'
     - Time
     - Total integration time of the observation
   * - detector_integration_time
     - str*
     - e.g. '0.01 d'
     - Time
     - Detector integration time; can not be smaller than 1 minute
   * - modulation_period
     - str*
     - e.g. '1 d'
     - Time
     - Duration to complete a modulation cycle of the array
   * - optimized_differential_output
     - int
     - e.g. 0
     - 1
     - Index corresponding to the :math:`n`\ th differential output of the array that the baselines should be optimized for
   * - optimized_star_separation
     - str/str*
     - 'habitable-zone'/e.g. '0.1 arcsec'
     - \-/Angle
     - Angular separation between the star and the planet that the baselines should be optimized for; 'habitable-zone' is also a valid input and will optimize for the habitable zone of the star
   * - optimized_wavelength
     - str*
     - e.g. '10 um'
     - Length
     - Wavelength that the baselines should be optimized for

\* String consisting of a number and a unit that can be parsed by astropy.units






Observatory
~~~~~~~~~~~~~~~~~~~
The ``instrument`` key contains several other keys that are used to configure the instrument. The key names, value types,
accepted values and meanings are given in the table below:

.. list-table:: Instrument Keys
   :widths: 26 7 20 7 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Dimension
     - Description
   * - array_configuration_matrix
     - ``sympy`` expression
     - valid ``sympy`` expression using symbols ``t``, ``tm`` and ``b``
     - \-
     - Array configuration matrix
   * - complex_amplitude_transfer_matrix
     - ``sympy`` expression
     - valid ``sympy`` expression
     - \-
     - Complex amplitude transfer matrix
   * - differential_outputs
     - list of tuples
     - e.g. [(2, 3)]
     - \-
     - List indicating the pair of tuples whose difference correspond to the indices of the differential outputs
   * - sep_at_max_mod_eff
     - list
     - e.g. [0.6]
     - \-
     - List indicating the separation in lambda/D at which the maximum modulation efficiency is achieved for the respective differential output
   * - aperture_diameter
     - str*
     - e.g. '2 m'
     - Length
     - Aperture diameter of the collectors
   * - baseline_ratio
     - int
     - e.g. 6
     - 1
     - Ratio of the imaging and the nulling baselines
   * - baseline_maximum
     - str*
     - e.g. '600 m'
     - Length
     - Maximum allowed baseline length
   * - baseline_minimum
     - str*
     - e.g. '10 m'
     - Length
     - Minimum allowed baseline length
   * - spectral_resolving_power
     - int
     - e.g. 100
     - 1
     - Spectral resolving power of the instrument
   * - wavelength_range_lower_limit
     - str*
     - e.g. '4 um'
     - Length
     - Lower limit of the wavelength range
   * - wavelength_range_upper_limit
     - str*
     - e.g. '18 um'
     - Length
     - Upper limit of the wavelength range
   * - throughput
     - float
     - e.g. 0.05
     - 1
     - Throughput of the unperturbed instrument
   * - quantum_efficiency
     - float
     - e.g. 0.7
     - 1
     - Quantum efficiency of the detector
   * - amplitude_perturbation > rms
     - str
     - e.g. '0.1 %'
     - %
     - RMS of the amplitude perturbation in percent
   * - amplitude_perturbation > color
     - str
     - 'white', pink', 'brown'
     - \-
     - Color of the power spectrum
   * - phase_perturbation > rms
     - str
     - e.g. '1 nm'
     - %
     - RMS of the phase perturbation
   * - phase_perturbation > color
     - str
     - 'white', pink', 'brown'
     - \-
     - Color of the power spectrum
   * - polarization_perturbation > rms
     - str
     - e.g. '0.001 rad'
     - %
     - RMS of the polarization perturbation in percent
   * - polarization_perturbation > color
     - str
     - 'white', pink', 'brown'
     - \-
     - Color of the power spectrum

\* String consisting of a number and a unit that can be parsed by astropy.units

Scene
~~~~~


*Star*


The *star* key contains several other keys that are used to define the star of the exoplanetary system. The key names,
value types, accepted values and meanings are given in the table below:

.. list-table:: Star Keys
   :widths: 26 7 20 7 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Dimension
     - Description
   * - name
     - str
     - e.g. 'Sun'
     - \-
     - Name of the star
   * - distance
     - str*
     - e.g. '10 pc'
     - Length
     - Distance to the star from the observatory
   * - mass
     - str*
     - e.g. '1 Msun'
     - Mass
     - Mass of the star
   * - radius
     - str*
     - e.g. '1 Rsun'
     - Length
     - Radius of the star
   * - temperature
     - str*
     - e.g. '5700 K'
     - Temperature
     - Effective temperature of the star
   * - luminosity
     - str*
     - e.g. '1 Lsun'
     - Watts
     - Luminosity of the star
   * - right_ascension
     - str*
     - e.g. '0 h'
     - Time (~Angle)
     - Right ascension of the star
   * - declination
     - str*
     - e.g. '-75 deg'
     - Angle
     - Declination of the star

\* String consisting of a number and a unit that can be parsed by astropy.units





*Exozodi*

The *exozodi* key contains several other keys that are used to define the zodi of the exoplanetary system (=exozodi).
The key names, value types, accepted values and meanings are given in the table below:

.. list-table:: Exozodi Keys
   :widths: 26 7 20 7 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Dimension
     - Description
   * - level
     - float
     - e.g. 3.0
     - 1
     - Amount of zodiacal dust in units of solar system zodi levels






Planets
~~~~~~~~~~~~~~~~~~~

The *planets* key contains a list of several other keys that are used to define the planets of the exoplanetary system.
The key names, value types, accepted values and meanings are given in the table below:

.. list-table:: Planets Keys
   :widths: 26 7 20 7 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Dimension
     - Description
   * - name
     - str
     - e.g. 'Earth'
     - \-
     - Name of the planet
   * - mass
     - str*
     - e.g. '1 Mearth'
     - Mass
     - Mass of the planet
   * - radius
     - str*
     - e.g. '1 Rearth'
     - Length
     - Radius of the planet
   * - temperature
     - str*
     - e.g. '300 K'
     - Temperature
     - Effective temperature of the planet
   * - semi_major_axis
     - str*
     - e.g. '1 au'
     - Length
     - Semi-major axis of the planet
   * - eccentricity
     - float
     - e.g. 0.0
     - 1
     - Eccentricity of the planet
   * - inclination
     - str*
     - e.g. '180 deg'
     - Angle
     - Inclination of the planet
   * - raan
     - str*
     - e.g. '0 deg'
     - Angle
     - Right ascension of the ascending node of the planet
   * - argument_of_periapsis
     - str*
     - e.g. '0 deg'
     - Angle
     - Argument of periapsis of the planet
   * - true_anomaly
     - str*
     - e.g. '0 deg'
     - Angle
     - True anomaly of the planet

\* String consisting of a number and a unit that can be parsed by astropy.units
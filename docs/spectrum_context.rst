.. _spectrum_context:

Spectrum Context File
=====================

Overview
--------

The spectrum context file is a YAML file that contains important context information about the input planet spectrum.
This context information is needed to properly interpret the input spectrum. A typical spectrum context file looks like
this:


.. include:: _static/spectrum_context.yaml
   :literal:

Note that the file has three main keys: `star`, `exozodi` and `planets`.

Keys
----

Star
~~~~~~~~~~~~~~~~~~~

The *star* key contains several other keys that are used to define the star of the exoplanetary system. The key names,
value types, accepted values and meanings are given in the table below:

.. list-table:: Star Keys
   :widths: 30 10 20 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Description
   * - name
     - str
     - e.g. Sun
     - Name of the star
   * - distance
     - str*
     - e.g. 10 pc
     - Distance to the star from the observatory
   * - mass
     - str*
     - e.g. 1 Msun
     - Mass of the star
   * - radius
     - str*
     - e.g. 1 Rsun
     - Radius of the star
   * - temperature
     - str*
     - e.g. 5700 K
     - Effective temperature of the star
   * - luminosity
     - str*
     - e.g. 1 Lsun
     - Luminosity of the star
   * - right_ascension
     - str*
     - e.g. 0 h
     - Right ascension of the star
   * - declination
     - str*
     - e.g. -75 deg
     - Declination of the star

\* String consisting of a number and a unit that can be parsed by astropy.units





Exozodi
~~~~~~~~~~~~~~~~~~~

The *exozodi* key contains several other keys that are used to define the zodi of the exoplanetary system (=exozodi).
The key names, value types, accepted values and meanings are given in the table below:

.. list-table:: Exozodi Keys
   :widths: 30 10 20 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Description
   * - level
     - float
     - e.g. 3.0
     - Amount of zodiacal dust in units of solar system zodi levels
   * - inclination
     - str*
     - e.g. 0 deg
     - Inclination of the exozodi

\* String consisting of a number and a unit that can be parsed by astropy.units





Planets
~~~~~~~~~~~~~~~~~~~

The *planets* key contains a list of several other keys that are used to define the planets of the exoplanetary system.
The key names, value types, accepted values and meanings are given in the table below:

.. list-table:: Planets Keys
   :widths: 30 10 20 40
   :header-rows: 1

   * - Name
     - Type
     - Accepted Values
     - Description
   * - name
     - str
     - e.g. Earth
     - Name of the planet
   * - mass
     - str*
     - e.g. 1 Mearth
     - Mass of the planet
   * - radius
     - str*
     - e.g. 1 Rearth
     - Radius of the planet
   * - temperature
     - str*
     - e.g. 300 K
     - Effective temperature of the planet
   * - semi_major_axis
     - str*
     - e.g. 1 au
     - Semi-major axis of the planet
   * - eccentricity
     - float
     - e.g. 0.0
     - Eccentricity of the planet
   * - inclination
     - str*
     - e.g. 180 deg
     - Inclination of the planet
   * - raan
     - str*
     - e.g. 0 deg
     - Right ascension of the ascending node of the planet
   * - argument_of_periapsis
     - str*
     - e.g. 0 deg
     - Argument of periapsis of the planet
   * - true_anomaly
     - str*
     - e.g. 0 deg
     - True anomaly of the planet

\* String consisting of a number and a unit that can be parsed by astropy.units






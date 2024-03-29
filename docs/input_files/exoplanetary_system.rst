.. _exoplanetary_system:

Exoplanetary System File
=====================

File Structure
---------------

The exoplanetary system file is a YAML file that contains information about the observed exoplanetary system.
This information is also important for properly interpreting the planetary input spectra, if they are provided. A
typical exoplanetary system file looks like this:

.. include:: ../_static/exoplanetary_system.yaml
   :literal:

Note that the file has three main keys: `star`, `exozodi` and `planets`.

.. note::
    Multiplanetary systems can simply be modeled by adding multiple planets to the `planets` list, i.e.

    ..  code-block:: yaml

        planets:
          - name: Earth
            mass: 1 Mearth
            radius: 1 Rearth
            temperature: 300 K
            semi_major_axis: 1 au
            eccentricity: 0.0
            inclination: 180 deg
            raan: 0 deg
            argument_of_periapsis: 0 deg
            true_anomaly: 0 deg
          - name: Mars
            mass: 0.1 Mearth
            radius: 0.5 Rearth
            temperature: 200 K
            semi_major_axis: 1.5 au
            eccentricity: 0.1
            inclination: 180 deg
            raan: 0 deg
            argument_of_periapsis: 0 deg
            true_anomaly: 0 deg

YAML Keys
---------------

Star
~~~~~~~~~~~~~~~~~~~

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
     - e.g. Sun
     - \-
     - Name of the star
   * - distance
     - str*
     - e.g. 10 pc
     - Length
     - Distance to the star from the observatory
   * - mass
     - str*
     - e.g. 1 Msun
     - Mass
     - Mass of the star
   * - radius
     - str*
     - e.g. 1 Rsun
     - Length
     - Radius of the star
   * - temperature
     - str*
     - e.g. 5700 K
     - Temperature
     - Effective temperature of the star
   * - luminosity
     - str*
     - e.g. 1 Lsun
     - Watts
     - Luminosity of the star
   * - right_ascension
     - str*
     - e.g. 0 h
     - Time (~Angle)
     - Right ascension of the star
   * - declination
     - str*
     - e.g. -75 deg
     - Angle
     - Declination of the star

\* String consisting of a number and a unit that can be parsed by astropy.units





Exozodi
~~~~~~~~~~~~~~~~~~~

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
   * - inclination
     - str*
     - e.g. 0 deg
     - Angle
     - Inclination of the exozodi

\* String consisting of a number and a unit that can be parsed by astropy.units





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
     - e.g. Earth
     - \-
     - Name of the planet
   * - mass
     - str*
     - e.g. 1 Mearth
     - Mass
     - Mass of the planet
   * - radius
     - str*
     - e.g. 1 Rearth
     - Length
     - Radius of the planet
   * - temperature
     - str*
     - e.g. 300 K
     - Temperature
     - Effective temperature of the planet
   * - semi_major_axis
     - str*
     - e.g. 1 au
     - Length
     - Semi-major axis of the planet
   * - eccentricity
     - float
     - e.g. 0.0
     - 1
     - Eccentricity of the planet
   * - inclination
     - str*
     - e.g. 180 deg
     - Angle
     - Inclination of the planet
   * - raan
     - str*
     - e.g. 0 deg
     - Angle
     - Right ascension of the ascending node of the planet
   * - argument_of_periapsis
     - str*
     - e.g. 0 deg
     - Angle
     - Argument of periapsis of the planet
   * - true_anomaly
     - str*
     - e.g. 0 deg
     - Angle
     - True anomaly of the planet

\* String consisting of a number and a unit that can be parsed by astropy.units






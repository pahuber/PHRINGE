# PHRINGE

[![PyPI](https://img.shields.io/pypi/v/phringe.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/phringe.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/phringe)][python version]
[![License](https://img.shields.io/pypi/l/phringe)][license]

[![Read the documentation at https://phringe.readthedocs.io/](https://img.shields.io/readthedocs/phringe/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/pahuber/phringe/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/pahuber/phringe/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/phringe/

[status]: https://pypi.org/project/phringe/

[python version]: https://pypi.org/project/phringe

[read the docs]: https://phringe.readthedocs.io/

[tests]: https://github.com/pahuber/phringe/actions?workflow=Tests

[codecov]: https://app.codecov.io/gh/pahuber/phringe

[pre-commit]: https://github.com/pre-commit/pre-commit

[black]: https://github.com/psf/black

`PHRINGE` is a **PH**otoelectron counts generato**R** for null**ING** int**E**rferometers capable of generating
synthetic data for space-based nulling interferometers. It can simulate the observation of an exoplanetary system and
generate realistic data in terms of photoelectron counts as a function of wavelength and time, considering both
astrophysical and instrumental noise sources.

## Documentation

The documentation including installation and usage instructions, examples and a general user documentation can be found
on [phringe.readthedocs.io](https://phringe.readthedocs.io/en/latest/).

## Features

- Symbolic input of complex amplitude transfer matrix and array positions, ensuring maximum flexibility in architecture modeling
- Symbolic calculation of instrument intensity response
- Noise models for astrophysical noise sources including stellar, local zodi and exozodi leakage
- Noise models for instrumental perturbations including amplitude, phase (OPD) and polarization rotation perturbations
- Export of synthetic data as a FITS file

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
PHRINGE is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz

[pypi]: https://pypi.org/

[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python

[file an issue]: https://github.com/pahuber/phringe/issues

[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/pahuber/phringe/blob/main/LICENSE

[contributor guide]: https://github.com/pahuber/phringe/blob/main/CONTRIBUTING.md

[command-line reference]: https://phringe.readthedocs.io/en/latest/usage.html

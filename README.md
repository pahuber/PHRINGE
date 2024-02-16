# SYGN

[![PyPI](https://img.shields.io/pypi/v/sygn.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/sygn.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/sygn)][python version]
[![License](https://img.shields.io/pypi/l/sygn)][license]

[![Read the documentation at https://sygn.readthedocs.io/](https://img.shields.io/readthedocs/sygn/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/pahuber/sygn/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/pahuber/sygn/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/sygn/
[status]: https://pypi.org/project/sygn/
[python version]: https://pypi.org/project/sygn
[read the docs]: https://sygn.readthedocs.io/
[tests]: https://github.com/pahuber/sygn/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/pahuber/sygn
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

`SYGN` is a **SY**nthetic photometry data **G**enerator for **N**ulling interferometers. It can model a variety of different interferometer architectures and account for astrophysical and instrumental noise sources. The documentation can be found on [sygn.readthedocs.io](https://sygn.readthedocs.io/en/latest/).

## Features

- Model different array architectures including different array configurations (Emma-X, Triangle, Pentagon) and different nulling schemes (double Bracewell, Kernel)
- Model noise contributions from astrophysical sources including stellar, local zodi and exozodi leakage
- Model noise contributions from systematic instrument perturbations including amplitude, phase (OPD) and polarization rotation perturbations
- Configure the osbervation and the observatory with all major parameters
- Configure the observed planetary system including the star, planets and exozodi
- Export the photometry data as a FITS file

## Requirements

As of version 0.0.3 `SYGN` requires the following packages:
- `click`
- `numpy`
- `astropy`
- `pydantic`
- `spectres`
- `poliastro`
- `tqdm`
- `numba`

## Installation

You can install `SYGN` via [pip] from [PyPI]:

```console
$ pip install sygn
```

## Usage

`SYGN` requires a configuration file and a system context file to run. The configuration file contains information about the simulation settings, the observation strategy and the observatory hardware (see [examples/config.yaml](./examples/config.yaml)). The system context file contains context information about the observed system, such as stellar and planetary bulk and orbital parameters (see [examples/system.yaml](./examples/system.yaml)).

To run `SYGN` open a terminal and write:
```console
$ sygn PATH_TO_CONFIG_FILE PATH_TO_SYSTEM_CONTEXT_FILE
```


Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SYGN_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/pahuber/sygn/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/pahuber/sygn/blob/main/LICENSE
[contributor guide]: https://github.com/pahuber/sygn/blob/main/CONTRIBUTING.md
[command-line reference]: https://sygn.readthedocs.io/en/latest/usage.html

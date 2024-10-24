from pathlib import Path

import click

from phringe.api import PHRINGE


@click.command()
@click.version_option()
@click.argument(
    'config',
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    '-s',
    '--seed',
    'seed',
    type=int,
    help="Seed to use for random number generation.",
    required=False
)
@click.option(
    '-g',
    '--gpu',
    'gpu',
    type=int,
    help="Indices of the GPUs to use.",
    multiple=True,
    required=False
)
@click.option(
    '-f',
    '--fits-suffix',
    'fits_suffix',
    type=str,
    help="Suffix of the FITS file name.",
    default='',
    required=False
)
@click.option('--fits/--no-fits', default=True, help="Write data to FITS file.")
@click.option('--copy/--no-copy', default=True, help="Write copy of input files to output directory.")
@click.option('--dir/--no-dir', default=True, help="Create a new directory in the output directory for each run.")
@click.option('--normalize/--no-normalize', default=False,
              help="Whether to normalize the data to unit RMS along the time axis.")
@click.option('--detailed/--no-detailed', default=False,
              help="Whether to run in detailed mode.")
def main(
        config: Path,
        seed: int = None,
        gpu: tuple = None,
        fits_suffix: str = '',
        fits: bool = True,
        copy: bool = True,
        dir: bool = True,
        normalize: bool = False,
        detailed: bool = False
):
    """PHRINGE. PHotoelectron counts generatoR for nullING intErferometers.

    CONFIG: Path to the configuration file.
    """
    phringe = PHRINGE()
    phringe.run(
        config_file_path=Path(config),
        seed=seed,
        gpu=gpu,
        fits_suffix=fits_suffix,
        write_fits=fits,
        create_copy=copy,
        create_directory=dir,
        normalize=normalize,
        detailed=detailed
    )

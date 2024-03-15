from pathlib import Path

import click

from phringe.phringe import PHRINGE


@click.command()
@click.version_option()
@click.argument(
    'config',
    type=click.Path(exists=True),
    required=True,
)
@click.argument(
    'exoplanetary_system',
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    '-s',
    '--spectrum',
    'spectrum_tuples',
    nargs=2,
    multiple=True,
    type=(str, click.Path(exists=True)),
    help="Planet name as specified in exoplanetary system file and path to the corresponding spectrum text file.",
    required=False
)
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(exists=True),
    help="Path to the output directory.",
    default=Path('.'),
    required=False
)
@click.option('--fits/--no-fits', default=True, help="Write data to FITS file.")
@click.option('--copy/--no-copy', default=True, help="Write copy of input files to output directory.")
@click.option('--sep/--no-sep', default=False,
              help="Generate separate data for the individual sources.")
def main(
        config: Path,
        exoplanetary_system: Path,
        spectrum_tuples=None,
        output_dir=Path('.'),
        fits=True,
        copy=True,
        sep=False
):
    """PHRINGE. synthetic PHotometRy data generator for nullING intErferometers.

    CONFIG: Path to the configuration file.
    EXOPLANETARY_SYSTEM: Path to the exoplanetary system file.
    """
    phringe = PHRINGE()
    phringe.run(
        config_file_path=config,
        exoplanetary_system_file_path=exoplanetary_system,
        spectrum_files=spectrum_tuples,
        output_dir=output_dir,
        write_fits=fits,
        create_copy=copy,
        generate_separate=sep
    )

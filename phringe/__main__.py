from pathlib import Path

import click

from phringe.api import API


@click.command()
@click.version_option()
@click.argument(
    'config_file_path',
    type=click.Path(exists=True),
    required=True,
)
@click.argument(
    'exoplanetary_system_file_path',
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    '-s',
    '--spectrum-file-path',
    'spectrum_file_path',
    type=click.Path(exists=True),
    help="Path to the spectrum text file.",
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
def main(
        config_file_path: Path,
        exoplanetary_system_file_path: Path,
        spectrum_file_path=None,
        output_dir=Path('.'),
        fits=True,
        copy=True
):
    """PHRINGE. synthetic PHotometRy data generator for nullING intErferometers.

    CONFIG_FILE_PATH: Path to the configuration file.
    EXOPLANETARY_SYSTEM_FILE_PATH: Path to the exoplanetary system file.
    """
    API.generate_data(
        config_file_path,
        exoplanetary_system_file_path,
        spectrum_file_path,
        output_dir,
        fits,
        copy
    )


if __name__ == "__main__":
    main(prog_name="PHRINGE")

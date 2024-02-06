"""Command-line interface."""
from pathlib import Path

import click

from src.sygn.core.entities.observation import Observation
from src.sygn.core.entities.observatory.observatory import Observatory
from src.sygn.core.entities.scene import Scene
from src.sygn.core.entities.settings import Settings
from src.sygn.core.processing.data_generator import DataGenerator
from src.sygn.io.fits_writer import FITSWriter
from src.sygn.io.txt_reader import TXTReader
from src.sygn.io.yaml_reader import YAMLReader


@click.command()
@click.version_option()
@click.argument('config_file_path', type=Path, required=True)
@click.argument('system_context_file_path', type=Path, required=True)
@click.option('--spectrum_file_path', '-s', type=Path, help="Path to the spectrum text file.", required=False)
def parse_arguments(config_file_path, system_context_file_path, spectrum_file_path=None) -> None:
    """SYGN. Generate synthetic photometry data for space-based nulling interferometers."""
    main(config_file_path, system_context_file_path, spectrum_file_path)


def main(config_file_path, system_context_file_path, spectrum_file_path=None) -> None:
    """SYGN. Generate synthetic photometry data for space-based nulling interferometers.
    """
    config_dict = YAMLReader().read(config_file_path)
    system_dict = YAMLReader().read(system_context_file_path)
    planet_spectrum = TXTReader().read(spectrum_file_path) if spectrum_file_path else None

    settings = Settings(**config_dict['settings'])
    observation = Observation(**config_dict['observation'])
    observatory = Observatory(**config_dict['observatory'])
    scene = Scene(**system_dict)

    settings.prepare(observation, observatory)
    observation.prepare()
    observatory.prepare(settings, observation)
    scene.prepare(settings, observatory, planet_spectrum)

    data_generator = DataGenerator(settings=settings, observation=observation, observatory=observatory, system=system)
    data = data_generator.run()

    fits_writer = FITSWriter()
    fits_writer.write(data)


if __name__ == "__main__":
    parse_arguments(prog_name="SYGN")

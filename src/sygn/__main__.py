"""Command-line interface."""
from pathlib import Path

import numpy as np
import click

from sygn.src.sygn.core.entities.observation import Observation
from sygn.src.sygn.core.entities.observatory.observatory import Observatory
from sygn.src.sygn.core.entities.settings import Settings
from sygn.src.sygn.core.entities.system import System
from sygn.src.sygn.core.processing.data_generator import DataGenerator
from sygn.src.sygn.io.fits_writer import FITSWriter
from sygn.src.sygn.io.txt_reader import TXTReader
from sygn.src.sygn.io.yaml_reader import YAMLReader


@click.command()
@click.version_option()
@click.option('--config', type=Path, help="Path to the configuration YAML file.", required=True)
@click.option('--system', type=Path, help="Path to the planetary system context YAML file.", required=True)
@click.option('--spectrum', type=Path, help="Path to the spectrum text file.", required=False)
def main(config, system, spectrum=None) -> None:
    """SYGN. Generate synthetic photometry data for space-based nulling interferometers."""
    config_dict = YAMLReader().read(config)
    system_dict = YAMLReader().read(system)
    spectrum = TXTReader().read(spectrum) if spectrum else None

    settings = Settings(**config_dict)
    observation = Observation(**config_dict)
    observatory = Observatory(**config_dict)
    system = System(**system_dict)

    settings.prepare(observation)
    observation.prepare()
    observatory.prepare(settings)
    system.prepare(settings, observatory, spectrum)

    data_generator = DataGenerator(settings=settings,
                                   observation=observation,
                                   observatory=observatory,
                                   system=system)
    data = data_generator.run()

    fits_writer = FITSWriter()
    fits_writer.write(data)


if __name__ == "__main__":
    main(prog_name="sygn")  # pragma: no cover

"""Command-line interface."""
from pathlib import Path

import click
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sygn.core.entities.observation import Observation
from sygn.core.entities.observatory.observatory import Observatory
from sygn.core.entities.scene import Scene
from sygn.core.entities.settings import Settings
from sygn.core.processing.data_generator import DataGenerator
from sygn.io.fits_writer import FITSWriter
from sygn.io.txt_reader import TXTReader
from sygn.io.yaml_reader import YAMLReader


@click.command()
@click.version_option()
@click.argument('config_file_path', type=click.Path(exists=True), required=True)
@click.argument('exoplanetary_system_file_path', type=click.Path(exists=True), required=True)
@click.option('--spectrum_file_path', '-s', type=click.Path(exists=True),
              help="Path to the spectrum text file.", required=False)
@click.option('--output_dir', '-o', type=click.Path(exists=True), help="Path to the output directory.",
              required=False)
def cli(config_file_path: Path, exoplanetary_system_file_path: Path, spectrum_file_path=None, output_dir: Path = None):
    """SYGN. Generate synthetic photometry data for space-based nulling interferometers."""
    main(config_file_path, exoplanetary_system_file_path, spectrum_file_path, output_dir)


def main(config_file_path, exoplanetary_system_file_path, spectrum_file_path=None, output_dir: Path = None):
    """Main function.

    :param config_file_path: Path to the configuration file
    :param exoplanetary_system_file_path: Path to the exoplanetary system file
    :param spectrum_file_path: Path to the spectrum file
    :param output_dir: Path to the output directory
    """
    config_dict = YAMLReader().read(config_file_path)
    system_dict = YAMLReader().read(exoplanetary_system_file_path)
    planet_spectrum = TXTReader().read(spectrum_file_path) if spectrum_file_path else None

    settings = Settings(**config_dict['settings'])
    observation = Observation(**config_dict['observation'])
    observatory = Observatory(**config_dict['observatory'])
    scene = Scene(**system_dict)

    settings.prepare(observation, observatory)
    observation.prepare()
    observatory.prepare(settings, observation, scene)
    scene.prepare(settings, observatory, planet_spectrum)

    data_generator = DataGenerator(settings=settings, observation=observation, observatory=observatory, scene=scene)
    data = data_generator.run()

    fits_writer = FITSWriter().write(data, output_dir)

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data[0].value, cmap='Greys')
    plt.suptitle('Differential Photon Counts', y=0.75, fontsize=14)
    plt.title('Planet + Astrophysical Noise + Instrumental Noise', fontsize=12)
    plt.ylabel('Spectral Channel')
    plt.xlabel('Time')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig('photon_counts3.jpg', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    cli(prog_name="SYGN")

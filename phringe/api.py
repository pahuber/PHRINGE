import shutil
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np

from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.core.processing.data_generator import DataGenerator
from phringe.io.fits_writer import FITSWriter
from phringe.io.utils import get_dict_from_path_or_dict, get_spectra_from_path
from phringe.io.yaml_handler import YAMLHandler


def main(
        config_file_path_or_dict: Union[Path, dict],
        exoplanetary_system_file_path_or_dict: Union[Path, dict],
        spectrum_tuple: tuple[tuple[str, Path]] = None,
        output_dir: Path = Path('.'),
        write_fits: bool = True,
        create_copy: bool = True,
        enable_stats: bool = False
) -> np.ndarray:
    """Generate synthetic photometry data.

    :param config_file_path_or_dict: The path to the configuration file or the configuration dictionary
    :param exoplanetary_system_file_path_or_dict: The path to the exoplanetary system file or the exoplanetary system dictionary
    :param spectrum_tuple: List of tuples containing the planet name and the path to the corresponding spectrum text file
    :param output_dir: The output directory
    :param write_fits: Whether to write the data to a FITS file
    :param create_copy: Whether to copy the input files to the output directory
    :param enable_stats: Whether to enable photon statistics by generating separate data sets for all sources
    :return: The data
    """
    config_dict = get_dict_from_path_or_dict(config_file_path_or_dict)
    system_dict = get_dict_from_path_or_dict(exoplanetary_system_file_path_or_dict)
    spectrum_tuple = get_spectra_from_path(spectrum_tuple) if spectrum_tuple else None
    output_dir = Path(output_dir)

    settings = Settings(**config_dict['settings'])
    observation = Observation(**config_dict['observation'])
    observatory = Observatory(**config_dict['observatory'])
    scene = Scene(
        **system_dict,
        wavelength_range_lower_limit=observatory.wavelength_range_lower_limit,
        wavelength_range_upper_limit=observatory.wavelength_range_upper_limit,
        spectrum_list=spectrum_tuple
    )

    settings.prepare(observation, observatory, scene)
    observation.prepare()
    observatory.prepare(settings, observation, scene)
    scene.prepare(settings, observation, observatory)

    data_generator = DataGenerator(settings, observation, observatory, scene, enable_stats=enable_stats)
    data = data_generator.run()

    if write_fits or create_copy:
        output_dir = output_dir.joinpath(f'out_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}')
        output_dir.mkdir(parents=True, exist_ok=True)

    if write_fits:
        if enable_stats:
            for source_name in data:
                fits_writer = FITSWriter().write(data[source_name], output_dir, source_name)
        else:
            fits_writer = FITSWriter().write(data, output_dir)

    if create_copy:
        if isinstance(config_file_path_or_dict, Path):
            shutil.copyfile(config_file_path_or_dict, output_dir.joinpath(config_file_path_or_dict.name))
        else:
            YAMLHandler().write(config_file_path_or_dict, output_dir.joinpath('config.yaml'))
        if isinstance(exoplanetary_system_file_path_or_dict, Path):
            shutil.copyfile(
                exoplanetary_system_file_path_or_dict,
                output_dir.joinpath(exoplanetary_system_file_path_or_dict.name)
            )
        else:
            YAMLHandler().write(exoplanetary_system_file_path_or_dict, output_dir.joinpath('system.yaml'))

    return data

import shutil
from datetime import datetime
from pathlib import Path

from sygn.core.entities.observation import Observation
from sygn.core.entities.observatory.observatory import Observatory
from sygn.core.entities.scene import Scene
from sygn.core.entities.settings import Settings
from sygn.core.processing.data_generator import DataGenerator
from sygn.io.fits_writer import FITSWriter
from sygn.io.txt_reader import TXTReader
from sygn.io.yaml_handler import YAMLHandler


class API:
    @staticmethod
    def generate_data(
            config_file_path_or_dict,
            exoplanetary_system_file_path_or_dict,
            spectrum_file_path=None,
            output_dir=Path('.'),
            fits=True,
            copy=True
    ):
        try:
            config_file_path_or_dict = Path(config_file_path_or_dict)
            config_dict = YAMLHandler().read(config_file_path_or_dict)
        except TypeError:
            config_dict = config_file_path_or_dict
        try:
            exoplanetary_system_file_path_or_dict = Path(exoplanetary_system_file_path_or_dict)
            system_dict = YAMLHandler().read(exoplanetary_system_file_path_or_dict)
        except TypeError:
            system_dict = exoplanetary_system_file_path_or_dict
        try:
            spectrum_file_path = Path(spectrum_file_path)
        except TypeError:
            pass
        try:
            output_dir = Path(output_dir)
            planet_spectrum = TXTReader().read(spectrum_file_path) if spectrum_file_path else None
        except TypeError:
            pass

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

        # If any output files should be written, create the data directory in the output directory
        if fits or copy:
            output_dir = output_dir.joinpath(f'out_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}')
            output_dir.mkdir(parents=True, exist_ok=True)

        if fits:
            fits_writer = FITSWriter().write(data, output_dir)

        if copy:
            if isinstance(config_file_path_or_dict, Path):
                shutil.copy(config_file_path_or_dict, output_dir.joinpath(config_file_path_or_dict.name))
            else:
                YAMLHandler().write(config_file_path_or_dict, output_dir.joinpath('config.yaml'))
            if isinstance(exoplanetary_system_file_path_or_dict, Path):
                shutil.copy(
                    exoplanetary_system_file_path_or_dict,
                    output_dir.joinpath(exoplanetary_system_file_path_or_dict.name)
                )
            else:
                YAMLHandler().write(exoplanetary_system_file_path_or_dict, output_dir.joinpath('system.yaml'))

        return data

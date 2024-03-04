import shutil
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
from astropy import units as u

from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.core.processing.data_generator import DataGenerator
from phringe.io.fits_writer import FITSWriter
from phringe.io.utils import get_dict_from_path, get_spectra_from_path
from phringe.io.yaml_handler import YAMLHandler


class PHRINGE():
    """Main class of PHRINGE.

    :param _data: The data
    :param _settings: The settings
    :param _observation: The observation
    :param _observatory: The observatory
    :param _scene: The scene
    """

    def get_data(self) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """Return the generated data.

        :return: The generated data
        """
        return self._data

    def get_wavelength_bin_centers(self) -> np.ndarray:
        """Return the wavelength bin centers.

        :return: The wavelength bin centers
        """
        return self._observatory.wavelength_bin_centers

    def get_time_steps(self) -> np.ndarray:
        """Return the observation time steps.

        :return: The observation time steps
        """
        return np.linspace(
            0,
            self._observation.total_integration_time,
            int(self._observation.total_integration_time / self._observation.detector_integration_time)
        ).to(u.s).value

    def run(
            self,
            config_file_path: Path,
            exoplanetary_system_file_path: Path,
            spectrum_tuple: tuple[tuple[str, Path]] = None,
            output_dir: Path = Path('.'),
            write_fits: bool = True,
            create_copy: bool = True,
            generate_separate: bool = False
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """Generate synthetic photometry data and return the total data as an array of shape N_differential_outputs x
        N_spectral_channels x N_time_steps or the data for each source separately in a dictionary of such arrays if
        enable_stats is True.

        :param config_file_path: The path to the configuration file
        :param exoplanetary_system_file_path: The path to the exoplanetary system file
        :param spectrum_tuple: List of tuples containing the planet name and the path to the corresponding spectrum text file
        :param output_dir: The output directory
        :param write_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to copy the input files to the output directory
        :param generate_separate: Whether to generate separate data sets for all individual sources
        :return: The data as an array or a dictionary of arrays if enable_stats is True
        """
        config_dict = get_dict_from_path(config_file_path)
        system_dict = get_dict_from_path(exoplanetary_system_file_path)
        self.run_with_dict(
            config_dict,
            system_dict,
            spectrum_tuple,
            output_dir,
            write_fits,
            create_copy,
            generate_separate,
            config_file_path,
            exoplanetary_system_file_path
        )

    def run_with_dict(
            self,
            config_dict: dict,
            exoplanetary_system_dict: dict,
            spectrum_tuple: tuple[tuple[str, Path]] = None,
            output_dir: Path = Path('.'),
            write_fits: bool = True,
            create_copy: bool = True,
            generate_separate: bool = False,
            config_file_path: Path = None,
            exoplanetary_system_file_path: Path = None
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """Generate synthetic photometry data and return the total data as an array of shape N_differential_outputs x
        N_spectral_channels x N_time_steps or the data for each source separately in a dictionary of such arrays if
        enable_stats is True. This method takes dictionaries as input instead of file paths.

        :param config_dict: The configuration dictionary
        :param exoplanetary_system_dict: The exoplanetary system dictionary
        :param spectrum_tuple: List of tuples containing the planet name and the path to the corresponding spectrum text file
        :param output_dir: The output directory
        :param write_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to copy the input files to the output directory
        :param generate_separate: Whether to generate separate data sets for all individual sources
        :param config_file_path: The path to the configuration file so the input config file can be copied if create_copy is True
        :param exoplanetary_system_file_path: The path to the exoplanetary system file so the input exoplanetary system file can be copied if create_copy is True

        :return: The data as an array or a dictionary of arrays if enable_stats is True
        """
        spectrum_tuple = get_spectra_from_path(spectrum_tuple) if spectrum_tuple else None
        output_dir = Path(output_dir)

        self._settings = Settings(**config_dict['settings'])
        self._observation = Observation(**config_dict['observation'])
        self._observatory = Observatory(**config_dict['observatory'])
        self._scene = Scene(
            **exoplanetary_system_dict,
            wavelength_range_lower_limit=self._observatory.wavelength_range_lower_limit,
            wavelength_range_upper_limit=self._observatory.wavelength_range_upper_limit,
            spectrum_list=spectrum_tuple
        )

        self._settings.prepare(self._observation, self._observatory, self._scene)
        self._observation.prepare()
        self._observatory.prepare(self._settings, self._observation, self._scene)
        self._scene.prepare(self._settings, self._observation, self._observatory)

        data_generator = DataGenerator(self._settings, self._observation, self._observatory, self._scene,
                                       generate_separate=generate_separate)
        # t0 = time.time_ns()
        self._data = data_generator.run()
        # t1 = time.time_ns()

        if write_fits or create_copy:
            output_dir = output_dir.joinpath(f'out_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}')
            output_dir.mkdir(parents=True, exist_ok=True)

        if write_fits:
            if generate_separate:
                for source_name in self._data:
                    fits_writer = FITSWriter().write(self._data[source_name], output_dir, source_name)
            else:
                fits_writer = FITSWriter().write(self._data, output_dir)

        if create_copy:
            if config_file_path:
                shutil.copyfile(config_file_path, output_dir.joinpath(config_file_path.name))
            else:
                YAMLHandler().write(config_file_path, output_dir.joinpath('config.yaml'))
            if exoplanetary_system_file_path:
                shutil.copyfile(
                    exoplanetary_system_file_path,
                    output_dir.joinpath(exoplanetary_system_file_path.name)
                )
            else:
                YAMLHandler().write(exoplanetary_system_file_path, output_dir.joinpath('system.yaml'))

        # print(f'PHRINGE run time: {(t1 - t0) / 1e9} seconds')

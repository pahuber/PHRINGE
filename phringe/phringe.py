import shutil
from datetime import datetime
from pathlib import Path
from typing import overload

from torch import Tensor

from phringe.core.director import Director
from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.scene import Scene
from phringe.core.entities.settings import Settings
from phringe.io.fits_writer import FITSWriter
from phringe.io.txt_reader import TXTReader
from phringe.io.utils import get_dict_from_path
from phringe.io.yaml_handler import YAMLHandler
from phringe.util.helpers import InputSpectrum


class PHRINGE():
    """Main class of PHRINGE.

    :param _data: The data
    :param _settings: The settings
    :param _observation: The observation
    :param _observatory: The observatory
    :param _scene: The scene
    """

    @staticmethod
    def _get_spectra_from_paths(spectrum_files: tuple[tuple[str, Path]]) -> list[InputSpectrum]:
        """Read the spectra from the paths and return a list of SpectrumContext objects.

        :param spectrum_files: List of tuples containing the planet name and the path to the corresponding spectrum text file
        :return: The spectra
        """
        try:
            input_spectra = []
            for index_path, (planet_name, spectrum_file_path) in enumerate(spectrum_files):
                input_spectra.append(InputSpectrum(planet_name, *TXTReader().read(Path(spectrum_file_path))))
        except TypeError:
            pass
        return input_spectra

    def get_data(self) -> Tensor:
        """Return the generated data.

        :return: The generated data
        """
        return self._director._data

    def get_wavelength_bin_centers(self) -> Tensor:
        """Return the wavelength bin centers.

        :return: The wavelength bin centers
        """
        return self._observatory.wavelength_bin_centers

    def get_time_steps(self) -> Tensor:
        """Return the observation time steps.

        :return: The observation time steps
        """
        return self._director._observatory_time_steps

    @overload
    def run(
            self,
            config_file_path: Path,
            exoplanetary_system_file_path: Path,
            spectrum_files: tuple[tuple[str, Path]] = None,
            gpus: tuple[int] = None,
            output_dir: Path = Path('.'),
            write_fits: bool = True,
            create_copy: bool = True
    ):
        ...

    @overload
    def run(
            self,
            settings: Settings,
            observatory: Observatory,
            observation: Observation,
            scene: Scene,
            spectrum_files: tuple[tuple[str, Path]] = None,
            gpus: tuple[int] = None,
            output_dir: Path = Path('.'),
            write_fits: bool = True,
            create_copy: bool = True
    ):
        ...

    def run(
            self,
            config_file_path: Path = None,
            exoplanetary_system_file_path: Path = None,
            settings: Settings = None,
            observatory: Observatory = None,
            observation: Observation = None,
            scene: Scene = None,
            spectrum_files: tuple[tuple[str, Path]] = None,
            gpus: tuple[int] = None,
            output_dir: Path = Path('.'),
            write_fits: bool = True,
            create_copy: bool = True,
    ):
        """Generate synthetic photometry data and return the total data as an array of shape N_differential_outputs x
        N_spectral_channels x N_time_steps or the data for each source separately in a dictionary of such arrays if
        enable_stats is True.

        :param config_file_path: The path to the configuration file
        :param exoplanetary_system_file_path: The path to the exoplanetary system file
        :param spectrum_files: List of tuples containing the planet name and the path to the corresponding spectrum text file
        :param gpus: Indices of the GPUs to use
        :param output_dir: The output directory
        :param write_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to copy the input files to the output directory
        :return: The data as an array or a dictionary of arrays if enable_stats is True
        """
        config_dict = get_dict_from_path(config_file_path) if config_file_path else None
        system_dict = get_dict_from_path(exoplanetary_system_file_path) if exoplanetary_system_file_path else None

        settings = Settings(**config_dict['settings']) if not settings else settings
        observatory = Observatory(**config_dict['observatory']) if not observatory else observatory
        observation = Observation(**config_dict['observation']) if not observation else observation
        scene = Scene(**system_dict) if not scene else scene
        input_spectra = PHRINGE._get_spectra_from_paths(spectrum_files) if spectrum_files else None

        self._director = Director(settings, observatory, observation, scene, input_spectra, gpus)
        self._director.run()

        if write_fits or create_copy:
            output_dir = output_dir.joinpath(f'out_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}')
            output_dir.mkdir(parents=True, exist_ok=True)

        if write_fits:
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

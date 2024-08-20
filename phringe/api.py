import pprint
import shutil
from datetime import datetime
from pathlib import Path
from typing import overload

import torch
from torch import Tensor

from phringe.core.director import Director
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation_mode import ObservationMode
from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.local_zodi import LocalZodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.scene import Scene
from phringe.core.entities.simulation import Simulation
from phringe.io.fits_writer import FITSWriter
from phringe.io.utils import load_config


class PHRINGE():
    """Main class of PHRINGE.
    """

    config = {
        'simulation': {
            'grid_size': None,
            'time_step_size': None,
            'has_planet_orbital_motion': None,
            'has_planet_signal': None,
            'has_stellar_leakage': None,
            'has_local_zodi_leakage': None,
            'has_exozodi_leakage': None,
            'has_amplitude_perturbations': None,
            'has_phase_perturbations': None,
            'has_polarization_perturbations': None,
        },
        'observation': {
            'solar_ecliptic_latitude': None,
            'total_integration_time': None,
            'detector_integration_time': None,
            'modulation_period': None,
            'optimized_differential_output': None,
            'optimized_star_separation': None,
            'optimized_wavelength': None,
        },
        'observatory': {
            'array_configuration_matrix': None,
            'complex_amplitude_transfer_matrix': None,
            'differential_outputs': None,
            'sep_at_max_mod_eff': None,
            'aperture_diameter': None,
            'baseline_ratio': None,
            'baseline_maximum': None,
            'baseline_minimum': None,
            'spectral_resolving_power': None,
            'wavelength_range_lower_limit': None,
            'wavelength_range_upper_limit': None,
            'throughput': None,
            'quantum_efficiency': None,
            'perturbations': {
                'amplitude_perturbation': {
                    'rms': None,
                    'color': None,
                },
                'phase_perturbation': {
                    'rms': None,
                    'color': None,
                },
                'polarization_perturbation': {
                    'rms': None,
                    'color': None,
                },
            }
        },
        'scene': {
            'star': {
                'name': None,
                'distance': None,
                'mass': None,
                'radius': None,
                'temperature': None,
                'luminosity': None,
                'right_ascension': None,
                'declination': None,
            },
            'exozodi': {
                'level': None
            },
            'planets': [
                {
                    None,
                },
            ],
        },
    }

    def get_data(self) -> Tensor:
        """Return the generated data.

        :return: The generated data
        """
        return self._director._data

    def get_field_of_view(self) -> Tensor:
        """Return the field of view.

        :return: The field of view
        """
        return self._director.field_of_view

    def get_intensity_response(self, source_name: str) -> Tensor:
        """Return the intensity response.

        :return: The intensity response
        """
        source = [source for source in self._director._sources if source.name == source_name][0]

        if isinstance(source, LocalZodi) or isinstance(source, Exozodi):
            sky_coordinates_x = source.sky_coordinates[0][:, None, :, :]
            sky_coordinates_y = source.sky_coordinates[1][:, None, :, :]
        elif isinstance(source, Planet) and self._director._has_planet_orbital_motion:
            sky_coordinates_x = source.sky_coordinates[0][None, :, :, :]
            sky_coordinates_y = source.sky_coordinates[1][None, :, :, :]
        else:
            sky_coordinates_x = source.sky_coordinates[0][None, None, :, :]
            sky_coordinates_y = source.sky_coordinates[1][None, None, :, :]

        num_in = self._director._number_of_inputs
        num_out = self._director._number_of_outputs
        time = self._director.simulation_time_steps[None, :, None, None]
        wavelength = self._director._wavelength_bin_centers[:, None, None, None]
        amplitude_pert = self._director.amplitude_pert_time_series
        phase_pert = self._director.phase_pert_time_series
        polarization_pert = self._director.polarization_pert_time_series

        return torch.stack([self._director._intensity_response[j](
            time,
            wavelength,
            sky_coordinates_x,
            sky_coordinates_y,
            torch.tensor(self._director._modulation_period),
            torch.tensor(self._director.nulling_baseline),
            *[self._director._amplitude for _ in range(num_in)],
            *[amplitude_pert[k][None, :, None, None] for k in range(num_in)],
            *[phase_pert[k][:, :, None, None] for k in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
            *[polarization_pert[k][None, :, None, None] for k in range(num_in)]
        ) for j in range(num_out)])

    def get_spectral_flux_density(self, source_name: str) -> Tensor:
        source = [source for source in self._director._sources if source.name == source_name][0]
        return source.spectral_flux_density

    def get_symbolic_intensity_response(self):
        """Return the intensity response.

        :return: The intensity response
        """
        return self._director._symbolic_intensity_response

    def get_template(self, time: Tensor, wavelength: Tensor, pos_x: Tensor, pos_y: Tensor, flux: Tensor) -> Tensor:
        """Return the template for a planet at position (pos_x, pos_y) in units of photoelectron counts.

        :return: The template
        """
        num_in = self._director._number_of_inputs
        time = time[None, :, None, None]
        wavelength = wavelength[:, None, None, None]

        diff_intensity_response = torch.stack([self._director._diff_intensity_response[i](
            time,
            wavelength,
            pos_x,
            pos_y,
            torch.tensor(self._director._modulation_period),
            torch.tensor(self._director.nulling_baseline),
            *[self._director._amplitude for _ in range(num_in)],
            *[torch.tensor(1) for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
        ) for i in range(len(self._director._differential_outputs))])

        diff_intensity_response = diff_intensity_response[:, :, :, 0, 0]

        return (flux[None, :, None] * diff_intensity_response * self._director._detector_integration_time
                * self._director._wavelength_bin_widths[None, :, None])

    def get_time_steps(self) -> Tensor:
        """Return the detector time steps.

        :return: The detector time steps
        """
        return self._director._detector_time_steps.cpu()

    def get_wavelength_bin_centers(self) -> Tensor:
        """Return the wavelength bin centers.

        :return: The wavelength bin centers
        """
        return self._director._wavelength_bin_centers.cpu()

    @overload
    def run(
            self,
            config_file_path: Path,
            gpu: int = None,
            fits_suffix: str = '',
            write_fits: bool = True,
            create_copy: bool = True,
            create_directory: bool = True,
            normalize: bool = False
    ):
        ...

    @overload
    def run(
            self,
            simulation: Simulation,
            instrument: Instrument,
            observation_mode: ObservationMode,
            scene: Scene,
            gpu: int = None,
            write_fits: bool = True,
            fits_suffix: str = '',
            create_copy: bool = True,
            create_directory: bool = True,
            normalize: bool = False
    ):
        ...

    def run(
            self,
            config_file_path: Path = None,
            simulation: Simulation = None,
            instrument: Instrument = None,
            observation_mode: ObservationMode = None,
            scene: Scene = None,
            gpu: int = None,
            fits_suffix: str = '',
            write_fits: bool = True,
            create_copy: bool = True,
            create_directory: bool = True,
            normalize: bool = False
    ):
        """Generate synthetic photometry data and return the total data as an array of shape N_diff_outputs x
        N_spec_channels x N_observation_time_steps.

        :param config_file_path: The path to the configuration file
        :param simulation: The simulation object
        :param instrument: The instrument object
        :param observation_mode: The observation mode object
        :param scene: The scene object
        :param gpu: Index of the GPU to use
        :param fits_suffix: The suffix for the FITS file
        :param write_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to copy the input files to the output directory
        :param create_directory: Whether to create a new directory in the output directory for each run
        :param normalize: Whether to normalize the data to unit RMS along the time axis
        :return: The data as an array or a dictionary of arrays if enable_stats is True
        """
        config_dict = load_config(config_file_path) if config_file_path else None

        simulation = Simulation(**config_dict['simulation']) if not simulation else simulation
        instrument = Instrument(**config_dict['instrument']) if not instrument else instrument
        observation_mode = ObservationMode(
            **config_dict['observation_mode']
        ) if not observation_mode else observation_mode
        scene = Scene(**config_dict['scene']) if not scene else scene

        self._director = Director(simulation, instrument, observation_mode, scene, gpu, normalize)

        self._director.run()

        if (write_fits or create_copy) and create_directory:
            output_dir = Path(f'out_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}')
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path('.')

        if write_fits:
            FITSWriter().write(self._director._data, output_dir, fits_suffix)

        if create_copy:
            if config_file_path:
                shutil.copyfile(config_file_path, output_dir.joinpath(config_file_path.name))
            else:
                dict_str = pprint.pformat(config_dict)
                file_content = f"config = {dict_str}\n"
                with open((output_dir.joinpath('config.py')), 'w') as file:
                    file.write(file_content)

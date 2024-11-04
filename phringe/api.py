import pprint
import shutil
from datetime import datetime
from pathlib import Path
from typing import overload, Union

import numpy as np
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

    def get_counts(self, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        """Return the counts.

        :return: The counts
        """
        if as_numpy:
            return self._director._counts.cpu().numpy()
        return self._director._counts

    def get_data(self, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        """Return the generated data.

        :return: The generated data
        """
        if as_numpy:
            return self._director._data.cpu().numpy()
        return self._director._data

    def get_field_of_view(self, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        """Return the field of view.

        :return: The field of view
        """
        if as_numpy:
            return self._director.field_of_view.cpu().numpy()
        return self._director.field_of_view

    def get_intensity_response_torch(self, source_name: str) -> Tensor:
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

        return torch.stack([self._director._intensity_response_torch[j](
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

    def get_spectral_flux_density(self, source_name: str, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        source = [source for source in self._director._sources if source.name == source_name][0]

        if as_numpy:
            return source.spectral_flux_density.cpu().numpy()
        return source.spectral_flux_density

    def get_symbolic_intensity_response(self):
        """Return the intensity response.

        :return: The intensity response
        """
        return self._director._symbolic_intensity_response

    def get_template_numpy(
            self,
            time: np.ndarray,
            wavelength_bin_center: np.ndarray,
            wavelength_bin_width: np.ndarray,
            pos_x: np.ndarray,
            pos_y: np.ndarray,
            flux: np.ndarray
    ):
        num_in = self._director._number_of_inputs

        # Ensure that the input tensors have the correct dimensions
        if time.size == 1:
            time = time[None, None, None, None]
        else:
            time = time[None, :, None, None]

        if wavelength_bin_center.size == 1:
            wavelength_bin_center = wavelength_bin_center[None, None, None, None]
        else:
            wavelength_bin_center = wavelength_bin_center[:, None, None, None]

        if wavelength_bin_width.size == 1:
            wavelength_bin_width = wavelength_bin_width[None, None, None, None, None]
        else:
            wavelength_bin_width = wavelength_bin_width[None, :, None, None, None]

        if pos_x.size == 1:
            pos_x = pos_x[None, None, None, None]
        else:
            pos_x = pos_x[None, None, :, :]

        if pos_y.size == 1:
            pos_y = pos_y[None, None, None, None]
        else:
            pos_y = pos_y[None, None, :, :]

        if flux.size == 1:
            flux = flux[None, None, None, None, None]
        else:
            flux = flux[None, :, None, None, None]

        amplitude = self._director._amplitude.cpu().numpy()

        diff_intensity_response = np.concatenate([self._director._diff_ir_numpy[i](
            time,
            wavelength_bin_center,
            pos_x,
            pos_y,
            self._director._modulation_period,
            self._director.nulling_baseline,
            *[amplitude for _ in range(num_in)],
            *[0 for _ in range(num_in)],
            *[0 for _ in range(num_in)],
            *[0 for _ in range(num_in)],
            *[0 for _ in range(num_in)],
        ) for i in range(len(self._director._differential_outputs))])

        return (
                flux
                * diff_intensity_response
                * self._director._detector_integration_time
                * wavelength_bin_width
        )

    def get_template_torch(
            self,
            time: Tensor,
            wavelength_bin_center: Tensor,
            wavelength_bin_width: Tensor,
            pos_x: Tensor,
            pos_y: Tensor,
            flux: Tensor
    ) -> Tensor:
        """Return the un-normalized template for a planet at position (pos_x, pos_y) in units of photoelectron counts.
        The dimension of the output is N_diff_outputs x N_lambda x N_time x N_pos_x x N_pos_y.

        :return: The template
        """
        num_in = self._director._number_of_inputs

        # Ensure that the input tensors have the correct dimensions
        if time.dim() == 0:
            time = time[None, None, None, None]
        elif time.dim() == 1:
            time = time[None, :, None, None]
        else:
            raise ValueError('Time must be a scalar or a 1D tensor.')

        if wavelength_bin_center.dim() == 0:
            wavelength_bin_center = wavelength_bin_center[None, None, None, None]
        elif wavelength_bin_center.dim() == 1:
            wavelength_bin_center = wavelength_bin_center[:, None, None, None]
        else:
            raise ValueError('Wavelength bin center must be a scalar or a 1D tensor.')

        if wavelength_bin_width.dim() == 0:
            wavelength_bin_width = wavelength_bin_width[None, None, None, None, None]
        elif wavelength_bin_width.dim() == 1:
            wavelength_bin_width = wavelength_bin_width[None, :, None, None, None]
        else:
            raise ValueError('Wavelength bin width must be a scalar or a 1D tensor.')

        if pos_x.dim() == 0:
            pos_x = pos_x[None, None, None, None]
        elif pos_x.dim() == 2:
            pos_x = pos_x[None, None, :, :]
        else:
            raise ValueError('Position x must be a scalar or a 2D tensor.')

        if pos_y.dim() == 0:
            pos_y = pos_y[None, None, None, None]
        elif pos_y.dim() == 2:
            pos_y = pos_y[None, None, :, :]
        else:
            raise ValueError('Position y must be a scalar or a 2D tensor.')

        if flux.dim() == 0:
            flux = flux[None, None, None, None, None]
        elif flux.dim() == 1:
            flux = flux[None, :, None, None, None]
        else:
            raise ValueError('Flux must be a scalar or a 1D tensor.')

        diff_intensity_response = torch.stack([self._director._diff_ir_torch[i](
            time,
            wavelength_bin_center,
            pos_x,
            pos_y,
            torch.tensor(self._director._modulation_period),
            torch.tensor(self._director.nulling_baseline),
            *[self._director._amplitude for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
            *[torch.tensor(0) for _ in range(num_in)],
        ) for i in range(len(self._director._differential_outputs))])

        return (
                flux
                * diff_intensity_response
                * self._director._detector_integration_time
                * wavelength_bin_width
        )

    def get_time_steps(self, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        """Return the detector time steps.

        :return: The detector time steps
        """
        if as_numpy:
            return self._director._detector_time_steps.cpu().numpy()
        return self._director._detector_time_steps

    def get_wavelength_bin_centers(self, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        """Return the wavelength bin centers.

        :return: The wavelength bin centers
        """
        if as_numpy:
            return self._director._wavelength_bin_centers.cpu().numpy()
        return self._director._wavelength_bin_centers

    def get_wavelength_bin_widths(self, as_numpy: bool) -> Union[np.ndarray, Tensor]:
        """Return the wavelength bin widths.

        :return: The wavelength bin widths
        """
        if as_numpy:
            return self._director._wavelength_bin_widths.cpu().numpy()
        return self._director._wavelength_bin_widths

    @overload
    def run(
            self,
            config_file_path: Path,
            seed: int = None,
            gpu: int = None,
            fits_suffix: str = '',
            write_fits: bool = True,
            create_copy: bool = True,
            create_directory: bool = True,
            normalize: bool = False,
            detailed: bool = False,
            extra_memory: int = 1
    ):
        ...

    @overload
    def run(
            self,
            simulation: Simulation,
            instrument: Instrument,
            observation_mode: ObservationMode,
            scene: Scene,
            seed: int = None,
            gpu: int = None,
            write_fits: bool = True,
            fits_suffix: str = '',
            create_copy: bool = True,
            create_directory: bool = True,
            normalize: bool = False,
            detailed: bool = False,
            extra_memory: int = 1
    ):
        ...

    def run(
            self,
            config_file_path: Path = None,
            simulation: Simulation = None,
            instrument: Instrument = None,
            observation_mode: ObservationMode = None,
            scene: Scene = None,
            seed: int = None,
            gpu: int = None,
            fits_suffix: str = '',
            write_fits: bool = True,
            create_copy: bool = True,
            create_directory: bool = True,
            normalize: bool = False,
            detailed: bool = False,
            extra_memory: int = 1
    ):
        """Generate synthetic photometry data and return the total data as an array of shape N_diff_outputs x
        N_spec_channels x N_observation_time_steps.

        :param config_file_path: The path to the configuration file
        :param simulation: The simulation object
        :param instrument: The instrument object
        :param observation_mode: The observation mode object
        :param scene: The scene object
        :param seed: The seed for the random number generator
        :param gpu: Index of the GPU to use
        :param fits_suffix: The suffix for the FITS file
        :param write_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to copy the input files to the output directory
        :param create_directory: Whether to create a new directory in the output directory for each run
        :param normalize: Whether to normalize the data to unit RMS along the time axis
        :param detailed: Whether to run in detailed mode, i.e. return all the interferometric outputs
        :param extra_memory: Factor to split the calculations into smaller chunks to save memory
        :return: The data as an array or a dictionary of arrays if enable_stats is True
        """
        config_dict = load_config(config_file_path) if config_file_path else None

        simulation = Simulation(**config_dict['simulation']) if not simulation else simulation
        instrument = Instrument(**config_dict['instrument']) if not instrument else instrument
        observation_mode = ObservationMode(
            **config_dict['observation_mode']
        ) if not observation_mode else observation_mode
        scene = Scene(**config_dict['scene']) if not scene else scene

        # If seed is None, set torch and numpy seeds to a random number to prevent the same seed from being used when
        # PHRINGE.run is called several times in a row
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._director = Director(
            simulation,
            instrument,
            observation_mode,
            scene,
            gpu,
            normalize,
            detailed,
            extra_memory
        )

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

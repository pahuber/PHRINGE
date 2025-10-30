from phringe.core.perturbations.power_law_psd_perturbation import PowerLawPSDPerturbation
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell
from phringe.util.baseline import OptimizedNullingBaseline

config = {
    'observation': {
        'solar_ecliptic_latitude': '0 deg',  # Used to determine the local zodi brightness
        'total_integration_time': '4 d',  # Total integration time of the observation
        'detector_integration_time': '0.02 d',  # Results in 200 time steps; use between 100 and 1000
        'modulation_period': '4 d',  # Period of the interferometer modulation, e.g. rotation
        'nulling_baseline': OptimizedNullingBaseline(  # Alternatively a fixed value, e.g. '10 m'
            angular_star_separation='habitable-zone',  # Alternatively a numerical value, e.g. '0.1 arcsec'
            wavelength='10 um',
            sep_at_max_mod_eff=DoubleBracewell.sep_at_max_mod_eff[0]
        ),
    },
    'instrument': {
        'array_configuration_matrix': XArrayConfiguration.acm,  # Array configuration; collector position and motion
        'complex_amplitude_transfer_matrix': DoubleBracewell.catm,  # Beam combiner transfer matrix
        'kernels': DoubleBracewell.kernels,  # Kernel matrix
        'aperture_diameter': '3.5 m',
        'nulling_baseline_max': '600 m',
        'nulling_baseline_min': '5 m',
        'spectral_resolving_power': 30,
        'wavelength_min': '4 um',
        'wavelength_max': '18.5 um',
        'wavelength_bands_boundaries': [],
        'throughput': 0.05,
        'quantum_efficiency': 0.7,
        'amplitude_perturbation': PowerLawPSDPerturbation(coefficient=1, rms='0.1%'),
        'phase_perturbation': PowerLawPSDPerturbation(coefficient=1, rms='1.5 nm', chromatic=True),
        'polarization_perturbation': PowerLawPSDPerturbation(coefficient=1, rms='0.001 rad'),
    },
    'scene': {
        'star': {
            'name': 'Sun',
            'distance': '10 pc',
            'mass': '1 Msun',
            'radius': '1 Rsun',
            'temperature': '5700 K',
            'right_ascension': '10 hourangle',
            'declination': '45 deg',
        },
        'exozodi': {
            'level': 3,
            # 'host_star_distance': '10 pc', # Only required if no star is present in the scene
            # 'host_star_luminosity': '1 Lsun', # Only required if no star is present in the scene
        },
        'local_zodi': {
            # 'host_star_right_ascension': '10 hourangle', # Only required if no star is present in the scene
            # 'host_star_declination': '45 deg',}, # Only required if no star is present in the scene
        },
        'planets': [
            {
                'name': 'Earth',
                'has_orbital_motion': False,
                'mass': '1 Mearth',
                'radius': '1 Rearth',
                'temperature': '254 K',
                'semi_major_axis': '1 au',
                'eccentricity': '0',
                'inclination': '0 deg',
                'raan': '0 deg',
                'argument_of_periapsis': '135 deg',
                'true_anomaly': '0 deg',
                'input_spectrum': None  # Will generate a blackbody spectrum, see tutorial for custom spectra
            },
            # Add more planets here
            # {
            #     'name': 'Mars',
            #     'mass': '1 Mearth',
            #     'radius': '1 Rearth',
            #     'temperature': '288 K',
            #     'semi_major_axis': '1 au',
            #     'eccentricity': '0',
            #     'inclination': '0 deg',
            #     'raan': '0 deg',
            #     'argument_of_periapsis': '45 deg',
            #     'true_anomaly': '0 deg',
            #     'input_spectrum': None
            # },
        ],
    },
}

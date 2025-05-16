from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewellBeamCombiner

config = {
    'observation': {
        'solar_ecliptic_latitude': '0 deg',
        'total_integration_time': '1 d',
        'detector_integration_time': '600 s',
        'modulation_period': '1 d',
        'optimized_differential_output': 0,
        'optimized_star_separation': 'habitable-zone',
        'optimized_wavelength': '10 um',
    },
    'instrument': {
        'array_configuration_matrix': XArrayConfiguration.acm,
        'complex_amplitude_transfer_matrix': DoubleBracewellBeamCombiner.catm,
        'differential_outputs': DoubleBracewellBeamCombiner.diff_out,
        'sep_at_max_mod_eff': DoubleBracewellBeamCombiner.sep_at_max_mod_eff,
        'aperture_diameter': '2 m',
        'baseline_maximum': '600 m',
        'baseline_minimum': '5 m',
        'spectral_resolving_power': 30,
        'wavelength_min': '4 um',
        'wavelength_max': '18.5 um',
        'wavelength_bands_boundaries': ['8 um'],
        'throughput': 0.05,
        'quantum_efficiency': 0.7,
        'perturbations': {
            'amplitude': {
                'rms': '0.1 %',
                'color': 'pink',
            },
            'phase': {
                'rms': '1.5 nm',
                'color': 'pink',
            },
            'polarization': {
                'rms': '0.001 rad',
                'color': 'pink',
            },
        }
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
            'level': 3
        },
        'local_zodi': {},
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
                'input_spectrum': None
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

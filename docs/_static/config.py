from sympy import Matrix, sin, exp, pi, I, cos, symbols, sqrt

########################################################################################################################
# Array Configuration Matrix (2 x N_in)
########################################################################################################################

t, tm, b = symbols('t tm b')  # Do not change this (t: time, tm: modulation period, b: baseline)

q = 6
acm = (b / 2
       * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                 [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
       * Matrix([[q, q, -q, -q],
                 [1, -1, -1, 1]]))

########################################################################################################################
# Complex Amplitude Transfer Matrix (N_out x N_in)
########################################################################################################################

catm = 1 / 2 * Matrix([[0, 0, sqrt(2), sqrt(2)],
                       [sqrt(2), sqrt(2), 0, 0],
                       [1, -1, -exp(I * pi / 2), exp(I * pi / 2)],
                       [1, -1, exp(I * pi / 2), -exp(I * pi / 2)]])

diff_out = [(2, 3)]
sep_at_max_mod_eff = [0.6]

# ep = exp(I * pi / 2)
# em = exp(-I * pi / 2)
#
# catm = 1 / 4 * Matrix([[2, 2, 2, 2],
#                        [1 + ep, 1 - ep, -1 + ep, -1 - ep],
#                        [1 - em, -1 - em, 1 + em, -1 + em],
#                        [1 + ep, 1 - ep, -1 - ep, -1 + ep],
#                        [1 - em, -1 - em, -1 + em, 1 + em],
#                        [1 + ep, -1 - ep, 1 - ep, -1 + ep],
#                        [1 - em, -1 + em, -1 - em, 1 + em]])
#
# diff_out = [(1, 2), (3, 4), (5, 6)]
# sep_at_max_mod_eff = [0.31, 1, 0.6]

########################################################################################################################
# Simulation, Observation Mode, Instrument and Scene
########################################################################################################################

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
        'array_configuration_matrix': acm,
        'complex_amplitude_transfer_matrix': catm,
        'differential_outputs': diff_out,
        'sep_at_max_mod_eff': sep_at_max_mod_eff,
        'aperture_diameter': '2 m',
        'baseline_maximum': '600 m',
        'baseline_minimum': '5 m',
        'spectral_resolving_power': 30,
        'wavelength_min': '4 um',
        'wavelength_max': '18.5 um',
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

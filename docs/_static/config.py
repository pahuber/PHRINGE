from sympy import Matrix, sqrt, sin, exp, pi, I, cos, symbols

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

########################################################################################################################
# Simulation, Observation Mode, Instrument and Scene
########################################################################################################################

config = {
    'simulation': {
        'grid_size': 100,
        'time_step_size': '1 d',
        'has_planet_orbital_motion': False,
        'has_planet_signal': True,
        'has_stellar_leakage': True,
        'has_local_zodi_leakage': True,
        'has_exozodi_leakage': True,
        'has_amplitude_perturbations': True,
        'has_phase_perturbations': True,
        'has_polarization_perturbations': True,
    },
    'observation_mode': {
        'solar_ecliptic_latitude': '0 deg',
        'total_integration_time': '100 d',
        'detector_integration_time': '1 d',
        'modulation_period': '100 d',
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
        'baseline_ratio': 6,
        'baseline_maximum': '600 m',
        'baseline_minimum': '5 m',
        'spectral_resolving_power': 20,
        'wavelength_range_lower_limit': '4 um',
        'wavelength_range_upper_limit': '18 um',
        'throughput': 0.05,
        'quantum_efficiency': 0.7,
        'perturbations': {
            'amplitude_perturbation': {
                'rms': '0.1 %',
                'color': 'pink',
            },
            'phase_perturbation': {
                'rms': '1 nm',
                'color': 'pink',
            },
            'polarization_perturbation': {
                'rms': '0.01 rad',
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
            'luminosity': '1 Lsun',
            'right_ascension': '0 h',
            'declination': '-75 deg',
        },
        'exozodi': {
            'level': 3
        },
        'planets': [
            {
                'name': 'Earth',
                'mass': '1 Mearth',
                'radius': '1 Rearth',
                'temperature': '288 K',
                'semi_major_axis': '1 au',
                'eccentricity': '0',
                'inclination': '180 deg',
                'raan': '0 deg',
                'argument_of_periapsis': '0 deg',
                'true_anomaly': '0 deg',
                'path_to_spectrum': 'spectrum.txt'
            },
        ],
    },
}

from phringe.core.perturbations.power_law_psd_perturbation import PowerLawPSDPerturbation
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.baseline import OptimalNullingBaseline
from phringe.lib.beam_combiner import DoubleBracewell

config = {
    'observation': {
        'solar_ecliptic_latitude': '0 deg',  # Used to determine the local zodi brightness
        'total_integration_time': '4 d',  # Total integration time of the observation
        'detector_integration_time': '0.04 d',  # Results in 200 time steps; it's recommended to use at least 100
        'modulation_period': '4 d',  # Period of the interferometer modulation, e.g., rotation around the line of sight
        'nulling_baseline': OptimalNullingBaseline(  # Alternatively, a fixed value, e.g., '10 m' can be set here
            angular_star_separation='habitable-zone',  # Alternatively a numerical value, e.g. '0.1 arcsec'
            wavelength='15 um',  # Wavelength the nulling baseline length is optimized for
            sep_at_max_mod_eff=DoubleBracewell.sep_at_max_mod_eff[0]  # This is an architecture-dependent constant
        ),
        'host_star_radius': '1 Rsun',  # Only required as a reference if no star is present in the scene
        'host_star_temperature': '5770 K',  # Only required as a reference if no star is present in the scene
        'host_star_mass': '1 Msun',  # Only required as a reference if no star is present in the scene
        'host_star_distance': '10 pc',  # Only required as a reference if no star is present in the scene
        'host_star_right_ascension': '10 hourangle',  # Only required as a reference if no star is present in the scene
        'host_star_declination': '45 deg',  # Only required as a reference if no star is present in the scene
    },
    'instrument': {
        'array_configuration_matrix': XArrayConfiguration.acm,  # Array configuration; collector position and motion
        'complex_amplitude_transfer_matrix': DoubleBracewell.catm,  # Beam combiner transfer matrix
        'kernels': DoubleBracewell.kernels,  # Kernel matrix
        'aperture_diameter': '3.5 m',
        'nulling_baseline_max': '600 m', # If the optimal nulling baseline is greater than this value, it will be set to this value
        'nulling_baseline_min': '25 m', # If the optimal nulling baseline is smaller than this value, it will be set to this value
        'spectral_resolving_power': 20,  # Usually at least 100 is recommended
        'wavelength_min': '4 um',
        'wavelength_max': '18.5 um',
        'wavelength_bands_boundaries': [],  # Use e.g. ['12 um'] to simulate two separate band passes from (4-12) um and (12-18.5) um
        'throughput': 0.12,  # Total throughput of the instrument
        'quantum_efficiency': 0.6,
        'amplitude_perturbation': PowerLawPSDPerturbation(coefficient=1, rms='0.1%'),  # Perturbations based on 1/f^c PSD, where f is the frequency and c is the coefficient
        'phase_perturbation': PowerLawPSDPerturbation(coefficient=1, rms='1.5 nm', chromatic=True),  # Perturbations based on 1/f^c PSD, where f is the frequency and c is the coefficient
        'polarization_perturbation': PowerLawPSDPerturbation(coefficient=1, rms='0.001 rad'),  # Perturbations based on 1/f^c PSD, where f is the frequency and c is the coefficient
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
            'level': 3,  # Zodi levels in units of Solar zodis
        },
        'local_zodi': {},
        'planets': [
            {
                'name': 'Earth',
                'propagate_orbit': False,  # If True, the planet's orbital motion will be simulated throughout the observation. If False, the planet will be fixed at the specified orbital parameters.
                'mass': '1 Mearth',
                'radius': '1 Rearth',
                'temperature': '254 K',
                'semi_major_axis': '1 au',
                'eccentricity': '0',
                'inclination': '0 deg',
                'raan': '0 deg',
                'argument_of_periapsis': '135 deg',
                'true_anomaly': '0 deg',
                'sed_loader': None  # If None, a blackbody spectrum will be generated based on the temperature. For custom spectra, see the dedicated tutorial.
            },
            # Add more planets here if necessary
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
            #     'sed_loader': None
            # },
        ],
    },
}

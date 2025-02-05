from sympy import Matrix, sin, exp, pi, I, cos, symbols, sqrt

from phringe.entities.instrument import Instrument

t, tm, b = symbols('t tm b')  # Do not change this (t: time, tm: modulation period, b: baseline)

q = 6
acm = (b / 2
       * Matrix(
            [[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
             [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]]
        )
       * Matrix(
            [[q, q, -q, -q],
             [1, -1, -1, 1]])
       )

catm = 1 / 2 * Matrix(
    [[0, 0, sqrt(2), sqrt(2)],
     [sqrt(2), sqrt(2), 0, 0],
     [1, -1, -exp(I * pi / 2), exp(I * pi / 2)],
     [1, -1, exp(I * pi / 2), -exp(I * pi / 2)]]
)

diff_out = [(2, 3)]
sep_at_max_mod_eff = [0.6]


class LIFEBaselineArchitecture(Instrument):

    def __init__(self):
        super().__init__(
            array_configuration_matrix=acm,
            complex_amplitude_transfer_matrix=catm,
            differential_outputs=diff_out,
            baseline_maximum='600 m',
            baseline_minimum='9 m',
            sep_at_max_mod_eff=sep_at_max_mod_eff,
            aperture_diameter='2 m',
            spectral_resolving_power='20',
            wavelength_min='4 um',
            wavelength_max='18.5 um',
            throughput=0.05,
            quantum_efficiency=0.7
        )

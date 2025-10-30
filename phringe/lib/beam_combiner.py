from sympy import Matrix, pi, sqrt, exp, I


class BaseBeamCombiner:
    catm = None
    kernels = None
    sep_at_max_mod_eff = None


class DoubleBracewell(BaseBeamCombiner):
    catm = 1 / 2 * Matrix([[0, 0, sqrt(2), sqrt(2)],
                           [sqrt(2), sqrt(2), 0, 0],
                           [1, -1, -exp(I * pi / 2), exp(I * pi / 2)],
                           [1, -1, exp(I * pi / 2), -exp(I * pi / 2)]])
    kernels = Matrix([[0, 0, 1, -1]])
    sep_at_max_mod_eff = [0.6]


class Kernel4(BaseBeamCombiner):
    ep = exp(1j * pi / 2)
    em = exp(-1j * pi / 2)

    catm = 1 / 4 * Matrix([[2, 2, 2, 2],
                           [1 + ep, 1 - ep, -1 + ep, -1 - ep],
                           [1 - em, -1 - em, 1 + em, -1 + em],
                           [1 + ep, 1 - ep, -1 - ep, -1 + ep],
                           [1 - em, -1 - em, -1 + em, 1 + em],
                           [1 + ep, -1 - ep, 1 - ep, -1 + ep],
                           [1 - em, -1 + em, -1 - em, 1 + em]])
    kernels = Matrix([[0, 1, -1, 0, 0, 0, 0],
                      [0, 0, 0, 1, -1, 0, 0],
                      [0, 0, 0, 0, 0, 1, -1]])
    sep_at_max_mod_eff = [0.4, 0.4, 0.4]


class Kernel5(BaseBeamCombiner):
    e2 = exp(2 * pi * I / 5)
    e4 = exp(4 * pi * I / 5)
    e6 = exp(6 * pi * I / 5)
    e8 = exp(8 * pi * I / 5)

    catm = 1 / sqrt(5) * Matrix([[1, 1, 1, 1, 1],
                                 [1, e2, e4, e6, e8],
                                 [1, e4, e8, e2, e6],
                                 [1, e6, e2, e8, e4],
                                 [1, e8, e6, e4, e2]])
    kernels = Matrix([[0, 1, 0, 0, -1],
                      [0, 0, 1, -1, 0]])
    sep_at_max_mod_eff = [2.68, 1.03]

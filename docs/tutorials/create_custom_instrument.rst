.. _create_custom_instrument:

Creating A Custom Instrument
============================

`PHRINGE` provides a couple predefined ``Instrument``s, but also allows the creation of custom instruments.
This requires the specification of the ``array_configuration_matrix``, the ``complex_amplitude_transfer_matrix``, the ``differential_outputs``, and
the ``sep_at_max_mod_eff`` arguments,

.. code-block:: Python

    inst = Instrument(
        array_configuration_matrix=acm,
        complex_amplitude_transfer_matrix=catm,
        differential_outputs=diff_outs,
        sep_at_max_mod_eff=sep_at_max_mod_eff
        # Other arguments
    )

as described in the :doc:`Instrument documentation <source/instrument>`.
This page explains how to specify these arguments.

Array Configuration Matrix
--------------------------
The array configuration matrix describes the time-dependent positions of the collector spacecraft.
It has the shape (2 x N_collectors); the two rows corresponding to the x and y coordinates. The matrix is usually a function
of time, ``t``, the modulation period of the array, ``tm^`` and the nulling baseline, ``b``.

For instance, the baseline design of LIFE features a rotating Emma-X configuration features 4 collectors arranged in an X-shape.
We can express this as

.. math::
    \mathbf{A}(t, t_m, b) = \frac{b}{2}\begin{bmatrix}
        \cos(2\pi t/t_m) & -\sin(2\pi t/t_m)\\
        \sin(2\pi t/t_m) & \cos(2\pi t/t_m)
    \end{bmatrix}
    \begin{bmatrix}
        q & q & -q & -q\\
        1 & -1 & -1 & 1
    \end{bmatrix},

where ``q = 6`` is the baseline ratio, i.e. the ratio between the imaging and the nulling baselines.

We define this equivalently in Python as

.. code-block:: python

    t, tm, b = symbols('t tm b')

    q = 6

    acm = (b / 2
           * Matrix([[cos(2 * pi / tm * t), -sin(2 * pi / tm * t)],
                     [sin(2 * pi / tm * t), cos(2 * pi / tm * t)]])
           * Matrix([[q, q, -q, -q],
                     [1, -1, -1, 1]]))



Beam Combination Transfer Matrix
--------------------------------
The beam combination transfer matrix has the shape (N_inputs x N_outputs), where N_inputs is equal to the number of collectors, N_collectors, and N_outputs is the number of outputs,
and describes how the input beams are combined to form the outputs. For instance, a dual Bracewell nuller with four inputs
and four outputs is described by

.. math::
    \mathbf{CATM} = \frac{1}{2}\begin{bmatrix}
        0 & 0 & \sqrt{2} & \sqrt{2}\\
        \sqrt{2} & \sqrt{2} & 0 & 0\\
        1 & -1 & -\exp(i \pi / 2) & \exp(i \pi / 2)\\
        1 & -1 & \exp(i \pi / 2) & -\exp(i \pi / 2)
    \end{bmatrix}.

We define this equivalently in Python as

.. code-block:: python

    catm = 1 / 2 * Matrix([[0, 0, sqrt(2), sqrt(2)],
                       [sqrt(2), sqrt(2), 0, 0],
                       [1, -1, -exp(I * pi / 2), exp(I * pi / 2)],
                       [1, -1, exp(I * pi / 2), -exp(I * pi / 2)]])

Differential Outputs
--------------------
The differential outputs are created by the difference of the outputs of the beam combiner.
Different beam combiners can have different numbers of differential outputs. They are specified in
`PHRINGE` as a list of tuples, containing the two indices of the outputs that are subtracted.
For instance, the dual Bracewell nuller has one differnetial output formed by the third and fourth outputs.
Thus:

.. code-block:: python

    diff_outs = [(2, 3)]

With indices starting at 0 in Python, this corresponds to the third and fourth outputs.

Separation at Maximum Modulation Efficiency
-------------------------------------------
The separation at maximum modulation efficiency describes the angular separation at which the instrument response modulates most efficiently and
is used to calculate the optimal baseline length. For instance, for a dual Bracewell nuller, the angular separation for which
the modulation efficiency is maximized is given by (`Dannert et al. 2022 <https://www.aanda.org/articles/aa/abs/2022/08/aa41958-21/aa41958-21.html>`_)

.. math::
    \theta_{\text{max}} \approx 0.6 \frac{\lambda}{b},

where :math:`\lambda` is the wavelength and :math:`b` is the nulling baseline.

We specify this in Python as a list containing the coefficient in front of the wavelength:

.. code-block:: python

    sep_at_max_mod_eff = [0.6]

Note that this list must contain a value for each differential output; they are usually different for different outputs.

.. note::
    These coefficients can be calculated by calculating the RMS of the intsrument response throughout the observation and plotting it as a function of angular separation.
    The maximum of this curve corresponds to the coefficient.



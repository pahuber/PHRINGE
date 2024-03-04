cimport cython
from libc.math cimport  cos, sin, sqrt
cimport numpy as np
import numpy as np

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)

cpdef double complex[:,::1] _calculate_complex_amplitude_base2(
        double wavelength,
        double phase_perturbation,
        double observatory_coordinates_x,
        double observatory_coordinates_y,
        double[:,::1] source_sky_coordinates_x,
        double[:,::1] source_sky_coordinates_y,
        unsigned int grid_size,
        double complex[:,::1] out):

    cdef double phase_term

    cdef unsigned int ix, iy
    cdef double factor = 2 * 3.1415926536 / wavelength

    for ix in range(grid_size):
        for iy in range(grid_size):



            phase_term = factor * (observatory_coordinates_y * source_sky_coordinates_x[ix,iy] +
                                                        observatory_coordinates_y * source_sky_coordinates_y[ix,iy] +
                                                        phase_perturbation)

            out[ix, iy] = (cos(phase_term) + 1j * sin(phase_term))
    return out


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double complex[:,:,:,:,::1] _calculate_complex_amplitude_base(
        double aperture_radius,
        double unperturbed_instrument_throughput,
        double[:,::1] amplitude_perturbation_time_series,
        double[:,::1] phase_perturbation_time_series,
        double[:,::1] observatory_coordinates_x,
        double[:,::1] observatory_coordinates_y,
        double[:,::1] source_sky_coordinates_x,
        double[:,::1] source_sky_coordinates_y,
        double[::1] inverse_wavelength_steps,
        unsigned int grid_size
):
    """Calculate the complex amplitude element for a single polarization.

    :param unperturbed_instrument_throughput: The unperturbed instrument throughput
    :param amplitude_perturbation_time_series: The amplitude perturbation time series
    :param phase_perturbation_time_series: The phase perturbation time series
    :param observatory_coordinates_x: The observatory x coordinates
    :param observatory_coordinates_y: The observatory y coordinates
    :param source_sky_coordinates_x: The source sky x coordinates
    :param source_sky_coordinates_y: The source sky y coordinates
    :return: The complex amplitude element
    """
    cdef double aperture_radius_ = aperture_radius
    cdef double unperturbed_instrument_throughput_ = unperturbed_instrument_throughput
    cdef double[:,:,:,:,:] amplitude_perturbation_time_series_ = amplitude_perturbation_time_series[None, ..., None, None]
    cdef double[:,:,:,:] phase_perturbation_time_series_ = phase_perturbation_time_series[..., None, None, None]
    cdef double[:,:,:,:] observatory_coordinates_x_ = observatory_coordinates_x[..., None, None]
    cdef double[:,:,:,:] observatory_coordinates_y_ = observatory_coordinates_y[..., None, None]
    cdef double[:,:,:,:] source_sky_coordinates_x_ = source_sky_coordinates_x[None, None, ...]
    cdef double[:,:,:,:] source_sky_coordinates_y_ = source_sky_coordinates_y[None, None, ...]
    cdef double[:,:,:,:,:] inverse_wavelength_steps_ = (inverse_wavelength_steps)[..., None, None, None, None]
    cdef unsigned int grid_size_ = grid_size

    cdef double[:,:,:,:] obs_x_source_x
    cdef double[:,:,:,:] obs_y_source_y
    cdef double[:,:,:,:] phase_pert
    cdef double[:,:,:,:] sum
    cdef double[:,:,:,:,:] exp
    cdef double complex[:,:,:,:,::1] a

    # cdef double[:,:,:,:,:] const = np.repeat(aperture_radius * sqrt(unperturbed_instrument_throughput), 5)
    # cdef const double complex[:,:,:,:,::1] exp_const = np.repeat(2j * 3.1415926536, 5)

    obs_x_source_x = (observatory_coordinates_x_ * source_sky_coordinates_x_)

    obs_y_source_y = (observatory_coordinates_y_ * source_sky_coordinates_y_)

    phase_pert = phase_perturbation_time_series_

    sum = obs_x_source_x + obs_y_source_y + phase_pert

    cdef double[:,:,:,:,:] sum2 = (sum)[None, ...]

    # cdef double[:,:,:,:,::1] exp = (exp_const * (1 / wavelength_steps)[..., None, None, None, None] *
    #        (obs_x_source_x + obs_y_source_y + phase_pert)[None, ...])
    exp = (inverse_wavelength_steps_ * sum2)

    a = amplitude_perturbation_time_series_ * (np.cos(exp) + 1j * np.sin(exp))

    return a
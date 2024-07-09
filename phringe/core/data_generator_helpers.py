import torch
from torch import Tensor


def calculate_complex_amplitude_base(
        amplitude_perturbation_time_series: Tensor,
        phase_perturbation_time_series: Tensor,
        observatory_coordinates_x: Tensor,
        observatory_coordinates_y: Tensor,
        source_sky_coordinates_x: Tensor,
        source_sky_coordinates_y: Tensor,
        wavelength_steps: Tensor,
) -> Tensor:
    """Calculate the complex amplitude element for a single polarization and return an array of shape N_wavelengths x
    N_collectors x N_time_steps x N_pix x N_pix.

    :param amplitude_perturbation_time_series: The amplitude perturbation time series
    :param phase_perturbation_time_series: The phase perturbation time series
    :param observatory_coordinates_x: The observatory x coordinates
    :param observatory_coordinates_y: The observatory y coordinates
    :param source_sky_coordinates_x: The source sky x coordinates
    :param source_sky_coordinates_y: The source sky y coordinates
    :param wavelength_steps: The wavelength steps
    :return: The complex amplitude element
    """
    exp_const = 2j * torch.pi

    # obs_x_source_x = torch.einsum('ij,kl->ijkl', observatory_coordinates_x, source_sky_coordinates_x)
    # obs_y_source_y = torch.einsum('ij,kl->ijkl', observatory_coordinates_y, source_sky_coordinates_y)

    # (Source = Star) or (source = Planet and no planet orbital motion)
    if len(source_sky_coordinates_x.shape) == 2:
        obs_x = observatory_coordinates_x[..., None, None]
        obs_y = observatory_coordinates_y[..., None, None]
        src_x = source_sky_coordinates_x[None, None, ...]
        src_y = source_sky_coordinates_y[None, None, ...]
    # Source = Planet and planet orbital motion
    elif len(source_sky_coordinates_x.shape) == 3 and observatory_coordinates_x.shape[-1] == \
            source_sky_coordinates_x.shape[0]:
        obs_x = observatory_coordinates_x[..., None, None]
        obs_y = observatory_coordinates_y[..., None, None]
        src_x = source_sky_coordinates_x[None, ...]
        src_y = source_sky_coordinates_y[None, ...]
    # Source = Local or Exozodi
    elif len(source_sky_coordinates_x.shape) == 3:
        obs_x = observatory_coordinates_x[None, ..., None, None]
        obs_y = observatory_coordinates_y[None, ..., None, None]
        src_x = source_sky_coordinates_x.unsqueeze(1).unsqueeze(2)
        src_y = source_sky_coordinates_y.unsqueeze(1).unsqueeze(2)

    obs_x_source_x = obs_x * src_x
    obs_y_source_y = obs_y * src_y

    phase_pert = phase_perturbation_time_series[..., None, None]

    if len(source_sky_coordinates_x.shape) == 2:
        sum = (obs_x_source_x + obs_y_source_y + phase_pert)[None, ...]
    else:
        sum = (obs_x_source_x + obs_y_source_y + phase_pert)
    # exp = exp_const * torch.einsum('i, jklm->ijklm', 1 / wavelength_steps, sum)

    exp = exp_const * (1 / wavelength_steps)[..., None, None, None, None] * sum

    a = amplitude_perturbation_time_series[None, ..., None, None] * torch.exp(exp)
    # a = torch.einsum('jk, ijklm->ijklm', amplitude_perturbation_time_series, torch.exp(exp))

    return a


def calculate_complex_amplitude(
        base_complex_amplitude: Tensor,
        polarization_perturbation_time_series: Tensor
) -> Tensor:
    """Calculate the complex amplitude and return an array of shape 2 x N_wavelengths x N_collectors x N_time_steps x
    N_pix x N_pix.

    :param base_complex_amplitude: The base complex amplitude
    :param polarization_perturbation_time_series: The polarization perturbation time series
    :return: The complex amplitude
    """
    complex_amplitude_x = base_complex_amplitude * torch.cos(
        polarization_perturbation_time_series[None, ..., None, None])

    complex_amplitude_y = base_complex_amplitude * torch.sin(
        polarization_perturbation_time_series[None, ..., None, None])

    return complex_amplitude_x, complex_amplitude_y


#
# def _calculate_intensity_response(
#         complex_amplitude_x: Tensor,
#         complex_amplitude_y: Tensor,
#         beam_combination_matrix: Tensor
# ) -> Tensor:
#     """Calculate the intensity response.
#
#     :param complex_amplitude_x: The complex amplitude x
#     :param complex_amplitude_y: The complex amplitude y
#     :param beam_combination_matrix: The beam combination matrix
#     :return: The intensity response
#     """
#     # i_x = torch.abs(torch.einsum('nj, ijklm->inklm', beam_combination_matrix, complex_amplitude_x)) ** 2
#     # i_y = torch.abs(torch.einsum('nj, ijklm->inklm', beam_combination_matrix, complex_amplitude_y)) ** 2
#
#     dot_product_x = beam_combination_matrix[None, ..., None, None, None] * complex_amplitude_x.unsqueeze(1)
#     result_x = torch.abs(torch.sum(dot_product_x, dim=2)) ** 2
#
#     dot_product_y = beam_combination_matrix[None, ..., None, None, None] * complex_amplitude_y.unsqueeze(1)
#     result_y = torch.abs(torch.sum(dot_product_y, dim=2)) ** 2
#
#     return result_x + result_y


def calculate_normalization(
        device: torch.device,
        source_sky_brightness_distribution: Tensor,
        simulation_wavelength_steps: Tensor
) -> Tensor:
    """Calculate the normalization of shape N_wavelengths.

    :param device: The device
    :param source_sky_brightness_distribution: The source sky brightness distribution
    :param simulation_wavelength_steps: The simulation wavelength steps
    :return: The normalization
    """
    if len(source_sky_brightness_distribution.shape) == 3:
        normalization = torch.empty(len(source_sky_brightness_distribution), device=device)
        for index_wavelength, wavelength in enumerate(simulation_wavelength_steps):
            source_sky_brightness_distribution2 = source_sky_brightness_distribution[index_wavelength]
            normalization[index_wavelength] = len(
                source_sky_brightness_distribution2[source_sky_brightness_distribution2 > 0]) if not len(
                source_sky_brightness_distribution2[source_sky_brightness_distribution2 > 0]) == 0 else 1
    elif len(source_sky_brightness_distribution.shape) == 4:
        normalization = torch.empty(
            (source_sky_brightness_distribution.shape[0], source_sky_brightness_distribution.shape[1]), device=device)

        for index_wavelength, wavelength in enumerate(simulation_wavelength_steps):
            for index_time in range(source_sky_brightness_distribution.shape[0]):
                source_sky_brightness_distribution2 = source_sky_brightness_distribution[index_time, index_wavelength]
                non_zero_pixels = len(
                    source_sky_brightness_distribution2[
                        source_sky_brightness_distribution2 > 0]
                )

                normalization[index_time, index_wavelength] = non_zero_pixels if (not non_zero_pixels == 0) and (
                    not torch.isnan(torch.tensor(non_zero_pixels))) else 1

    return normalization


def calculate_photon_counts_from_intensity_response(
        device: torch.device,
        intensity_response: Tensor,
        source_sky_brightness_distribution: Tensor,
        simulation_wavelength_steps: Tensor,
        simulation_wavelength_bin_widths: Tensor,
        simulation_time_step_duration: float,
        normalization: Tensor
) -> Tensor:
    """Calculate the photon counts and return an array of shape N_outputs x N_wavelengths x N_time_steps.

    :param intensity_response: The intensity response
    :param source_sky_brightness_distribution: The source sky brightness distribution
    :param simulation_wavelength_bin_widths: The simulation wavelength bin widths
    :param simulation_time_step_duration: The simulation time step duration
    :return: The photon counts
    """

    normalization = calculate_normalization(
        device,
        source_sky_brightness_distribution,
        simulation_wavelength_steps
    )

    c = 0

    # a = (simulation_time_step_duration * intensity_response.swapaxes(0, 2)
    #      * source_sky_brightness_distribution[None, None, ...]
    #      * simulation_wavelength_bin_widths[None, None, ..., None, None]
    #      / normalization[None, None, ..., None, None])

    if len(source_sky_brightness_distribution.shape) == 3:
        a = simulation_time_step_duration * torch.einsum(
            'ijklm, ilm, i, i-> ijklm',
            intensity_response,
            source_sky_brightness_distribution,
            simulation_wavelength_bin_widths,
            1 / normalization
        )
    elif len(source_sky_brightness_distribution.shape) == 4:
        a = simulation_time_step_duration * torch.einsum(
            'ijklm, kilm, i, ki-> ijklm',
            intensity_response,
            source_sky_brightness_distribution,
            simulation_wavelength_bin_widths,
            1 / normalization
        )
        mean_photon_counts = torch.sum(a, axis=(3, 4)).swapaxes(0, 1)

        return torch.poisson(mean_photon_counts)

    mean_photon_counts = torch.sum(a, axis=(3, 4)).swapaxes(0, 1)

    return torch.poisson(mean_photon_counts)

#
# def calculate_photon_counts(
#         device: torch.device,
#         aperture_radius: Tensor,
#         unperturbed_instrument_throughput: Tensor,
#         amplitude_perturbation_time_series: Tensor,
#         phase_perturbation_time_series: Tensor,
#         polarization_perturbation_time_series: Tensor,
#         observatory_coordinates_x: Tensor,
#         observatory_coordinates_y: Tensor,
#         source_sky_coordinates_x: Tensor,
#         source_sky_coordinates_y: Tensor,
#         source_sky_brightness_distribution: Tensor,
#         simulation_wavelength_steps: Tensor,
#         simulation_wavelength_bin_widths: Tensor,
#         simulation_time_step_duration: float,
#         beam_combination_matrix: Tensor,
#         normalization_array: Tensor
# ):
#     """Calculate the photon counts.
#
#     :param device: The device
#     :param aperture_radius: The aperture radius
#     :param unperturbed_instrument_throughput: The unperturbed instrument throughput
#     :param amplitude_perturbation_time_series: The amplitude perturbation time series
#     :param phase_perturbation_time_series: The phase perturbation time series
#     :param polarization_perturbation_time_series: The polarization perturbation time series
#     :param observatory_coordinates_x: The observatory x coordinates
#     :param observatory_coordinates_y: The observatory y coordinates
#     :param source_sky_coordinates_x: The source sky x coordinates
#     :param source_sky_coordinates_y: The source sky y coordinates
#     :param source_sky_brightness_distribution: The source sky brightness distribution
#     :param simulation_wavelength_steps: The simulation wavelength steps
#     :param simulation_wavelength_bin_widths: The simulation wavelength bin widths
#     :param simulation_time_step_duration: The simulation time step duration
#     :param beam_combination_matrix: The beam combination matrix
#     :return: The photon counts
#     """
#     base_complex_amplitude = _calculate_complex_amplitude_base(
#         amplitude_perturbation_time_series,
#         phase_perturbation_time_series,
#         observatory_coordinates_x,
#         observatory_coordinates_y,
#         source_sky_coordinates_x,
#         source_sky_coordinates_y,
#         simulation_wavelength_steps
#     ) * aperture_radius * torch.sqrt(unperturbed_instrument_throughput)
#
#     complex_amplitude_x, complex_amplitude_y = _calculate_complex_amplitude(
#         base_complex_amplitude,
#         polarization_perturbation_time_series
#     )
#
#     del base_complex_amplitude
#     del observatory_coordinates_x
#     del observatory_coordinates_y
#     gc.collect()
#
#     intensity_response = _calculate_intensity_response(
#         complex_amplitude_x,
#         complex_amplitude_y,
#         beam_combination_matrix
#     )
#     print('here')
#
#     photon_counts = _calculate_photon_counts_from_intensity_response(
#         device,
#         intensity_response,
#         source_sky_brightness_distribution,
#         simulation_wavelength_steps,
#         simulation_wavelength_bin_widths,
#         simulation_time_step_duration,
#         normalization_array
#     )
#     return photon_counts

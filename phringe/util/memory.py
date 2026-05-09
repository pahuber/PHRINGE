from typing import Generator

import psutil
import torch


def get_available_memory(device: torch.device) -> int:
    """Get the available memory in bytes.

    Parameters
    ----------
    device : torch.device
        The device

    Returns
    -------
    int
        The available memory in bytes.
    """
    if device.type == 'cuda':
        gpu_id = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        total_memory = gpu_props.total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        cached_memory = torch.cuda.memory_reserved(gpu_id)
        avail = total_memory - allocated_memory - cached_memory
    else:
        avail = psutil.virtual_memory().available

    return avail


def get_device(gpu: int) -> torch.device:
    """Get the device.

    Parameters
    ----------
    gpu : int
        The GPU number

    Returns
    -------
    torch.device
        The device.
    """
    if gpu and torch.cuda.is_available() and torch.cuda.device_count():
        if torch.max(torch.asarray(gpu)) > torch.cuda.device_count():
            raise ValueError(f'GPU number {torch.max(torch.asarray(gpu))} is not available on this machine.')
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    return device


def get_time_slices(
        n_time_steps: int,
        n_wavelengths: int,
        n_out: int,
        n_grid: int,
        device: torch.device,
        extra_memory: float
) -> torch.Tensor:
    """Get the time slices.

    Parameters
    ----------
    n_time_steps : int
        The number of time steps.
    n_wavelengths : int
        The number of wavelengths.
    n_out : int
        The number of outputs.
    n_grid : int
        The grid size.
    device : torch.device
        The device.
    extra_memory : float
        The extra memory factor.

    Returns
    -------
    torch.Tensor
        The time slices.
    """
    bytes_per_element = 4  # For float32
    overhead_factor = 4.0  # Empirical safety factor

    # Dominant tensor in counts calculation is of shape (n_out, n_w, n_t_slice, n_pix, n_pix)
    bytes_per_time_step = (
            n_out * n_wavelengths * n_grid * n_grid * bytes_per_element * overhead_factor
    )

    usable_memory = get_available_memory(device) / extra_memory * 0.9

    chunk_size = max(1, int(usable_memory // bytes_per_time_step))

    time_step_indices = torch.arange(0, n_time_steps + 1, chunk_size)

    if time_step_indices[-1] != n_time_steps:
        time_step_indices = torch.cat(
            (time_step_indices, torch.tensor([n_time_steps], device=time_step_indices.device))
        )

    return time_step_indices


def iter_time_slices(
        n_time_steps: int,
        n_wavelengths: int,
        n_out: int,
        n_grid: int,
        device: torch.device,
        extra_memory: float
) -> Generator:
    """Iterate over the time slices.

    Parameters
    ----------
    n_time_steps : int
        The number of time steps.
    n_wavelengths : int
        The number of wavelengths.
    n_out : int
        The number of outputs.
    n_grid : int
        The grid size.
    device : torch.device
        The device.
    extra_memory : float
        The extra memory factor.

    Returns
    -------
    Generator
        The time slices.
    """
    time_step_indices = get_time_slices(n_time_steps, n_wavelengths, n_out, n_grid, device, extra_memory)
    for i in range(len(time_step_indices) - 1):
        yield time_step_indices[i].item(), time_step_indices[i + 1].item()

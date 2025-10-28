import psutil
import torch


def get_available_memory(device: torch.device) -> int:
    """Get the available memory in bytes.

    :return: The available memory in bytes
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


import torch


def get_device(gpu: int) -> torch.device:
    """Get the device.

    :param gpu: The GPU
    :return: The device
    """
    if gpu and torch.cuda.is_available() and torch.cuda.device_count():
        if torch.max(torch.asarray(gpu)) > torch.cuda.device_count():
            raise ValueError(f'GPU number {torch.max(torch.asarray(gpu))} is not available on this machine.')
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    return device

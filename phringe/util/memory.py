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

import numpy as np
import torch


def _get_device(gpu: int) -> torch.device:
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


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
